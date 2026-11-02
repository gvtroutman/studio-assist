"""Bounded stdlib transport and durable local takes. No speech model imports."""
import base64
import hashlib
import io
import json
import math
import os
import shutil
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

from .catalog import ENGINES, MAX_AUDIO, MAX_REFERENCE, PROTOCOL, VoiceError, validate


def base_url(value):
    value = str(value).strip().rstrip('/')
    try:
        p = urllib.parse.urlsplit(value)
        valid = p.scheme in ('http', 'https') and p.hostname and not (p.username or p.password or p.query or p.fragment) and p.path in ('', '/')
        if p.port is not None and p.port == 0:
            valid = False
    except ValueError:
        valid = False
    if not valid:
        raise VoiceError('Enter the remote voice server URL, for example http://100.127.17.38:17494.')
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def reference_wav(path):
    try:
        if os.path.getsize(path) > MAX_REFERENCE:
            raise VoiceError('Use a reference WAV under 8 MB.')
        with open(path, 'rb') as f:
            data = f.read(MAX_REFERENCE + 1)
        check_wav(data, max_seconds=30)
    except OSError:
        raise VoiceError('The reference WAV cannot be opened.') from None
    return data


def check_wav(data, max_seconds=1200, *, analyze=True):
    try:
        with wave.open(io.BytesIO(data)) as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            duration = frames / rate
            if channels not in (1, 2) or width != 2 or not 8000 <= rate <= 192000 or not 0 < duration <= max_seconds:
                raise VoiceError('Use nonempty 16-bit PCM WAV audio, at most %s seconds.' % max_seconds)
            pcm = w.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise VoiceError('The WAV is truncated.')
    except (wave.Error, EOFError, ZeroDivisionError):
        raise VoiceError('The server/reference did not provide a valid 16-bit PCM WAV.') from None
    basic = dict(duration=round(duration, 4), sample_rate=rate, channels=channels)
    if not analyze:
        return basic
    samples = struct.iter_unpack('<h', pcm)
    peak = squares = count = clipped = 0
    for (sample,) in samples:
        absolute = abs(sample)
        peak = max(peak, absolute)
        squares += sample * sample
        clipped += absolute >= 32767
        count += 1
    return dict(basic,
                peak_db=round(20 * math.log10(peak / 32768), 2) if peak else None,
                rms_db=round(10 * math.log10(squares / count / (32768**2)), 2) if squares else None,
                clipped_samples=clipped, samples=count)


class Client:
    def __init__(self, url, key='', opener=None):
        self.url = base_url(url)
        self.key = key
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, route, payload=None, *, audio=False):
        headers = {'Accept': 'audio/wav' if audio else 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode('utf8')
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(self.url + '/v1' + route, data=data, headers=headers)
        limit = MAX_AUDIO if audio else 256 * 1024
        try:
            with self.opener.open(req, timeout=15) as response:
                raw = response.read(limit + 1)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise VoiceError('The voice server refused access. Check its server token.') from None
            detail = ''
            try:
                detail = json.loads(error.read(4096)).get('error', '')
            except (ValueError, OSError, AttributeError):
                pass
            raise VoiceError('Voice server HTTP %s%s' % (error.code, ': ' + str(detail)[:400] if detail else '')) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise VoiceError('The remote voice server did not answer. Check the LLM PC, URL and server. A submitted job may still be running; no automatic resubmission was made.') from None
        if len(raw) > limit:
            raise VoiceError('The voice server returned too much data.')
        if audio:
            check_wav(raw, analyze=False)
            return raw
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            raise VoiceError('The voice server returned an invalid response.') from None
        if not isinstance(result, dict):
            raise VoiceError('The voice server response must be an object.')
        return result

    def capabilities(self):
        result = self.request('/capabilities')
        if result.get('protocol') != PROTOCOL or not isinstance(result.get('engines'), list):
            raise VoiceError('This endpoint is not a compatible Studio Assist voice server.')
        return result

    def generate(self, request, *, ident=None, reference=None, cancel=None, progress=None, timeout=1800):
        request = validate(request, reference=bool(reference))
        ident = ident or uuid.uuid4().hex
        payload = dict(request, id=ident)
        if reference and ENGINES[request['engine']]['reference'] != 'unused':
            check_wav(reference, max_seconds=30)
            payload['reference'] = base64.b64encode(reference).decode('ascii')
        if cancel and cancel.is_set():
            raise VoiceError('Voice generation cancelled.')
        result = self.request('/jobs', payload)
        if result.get('id') != ident:
            raise VoiceError('The voice server returned the wrong job ID.')
        return self.collect(ident, request, cancel=cancel, progress=progress, timeout=timeout)

    def collect(self, ident, request, *, cancel=None, progress=None, timeout=1800):
        """Recover an existing ID without ever submitting or duplicating work."""
        if not isinstance(ident, str) or len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise VoiceError('Invalid remote job ID.')
        deadline = time.monotonic() + timeout
        sent_cancel = False
        while True:
            if cancel and cancel.is_set() and not sent_cancel:
                self.request('/jobs/' + ident + '/cancel', {})
                sent_cancel = True
                deadline = min(deadline, time.monotonic() + 30)
            result = self.request('/jobs/' + ident)
            if result.get('id') != ident:
                raise VoiceError('The voice server returned the wrong job status.')
            status = result.get('status')
            if progress:
                progress('%s · %s' % (ENGINES[request['engine']]['name'], result.get('progress') or status))
            if status in ('cancelled', 'failed'):
                raise VoiceError(result.get('error') or 'Voice generation ' + status)
            if status == 'completed':
                if sent_cancel or (cancel and cancel.is_set()):
                    raise VoiceError('Voice generation cancelled.')
                return self.request('/jobs/' + ident + '/audio', audio=True), result
            if status not in ('queued', 'running', 'cancelling'):
                raise VoiceError('The voice server returned an unknown job state.')
            if time.monotonic() >= deadline:
                if not sent_cancel:
                    self.request('/jobs/' + ident + '/cancel', {})
                raise VoiceError('Voice job timed out. Cancellation was requested; check the remote job before retrying.')
            # An Event wait makes Stop responsive without another polling thread.
            if cancel and not sent_cancel:
                cancel.wait(.5)
            else:
                time.sleep(.5)


class Library:
    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.path = os.path.join(self.directory, 'library.json')
        self.data = {'draft': {}, 'profiles': [], 'takes': []}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding='utf8') as f:
                    loaded = json.load(f)
                self.check_data(loaded)
                self.data.update(loaded)
            except (OSError, ValueError):
                raise VoiceError('The Voice Studio library could not be read. Keep it for recovery: ' + self.path) from None

    @staticmethod
    def check_data(data):
        """Reject malformed saved records before any GUI code consumes them."""
        def ident(value):
            return isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value)
        if not isinstance(data, dict) or not isinstance(data.get('draft', {}), dict):
            raise ValueError()
        if any(not isinstance(v, str) for v in data.get('draft', {}).values()):
            raise ValueError()
        for key in ('profiles', 'takes', 'pending'):
            if not isinstance(data.get(key, []), list) or any(not isinstance(r, dict) for r in data.get(key, [])):
                raise ValueError()
        for row in data.get('profiles', []):
            if not isinstance(row.get('name'), str) or any(not isinstance(v, str) for v in row.values()):
                raise ValueError()
        seen = set()
        for row in data.get('takes', []):
            if not ident(row.get('id')) or row['id'] in seen or row.get('engine') not in ENGINES:
                raise ValueError()
            seen.add(row['id'])
            if any(not isinstance(row.get(k, ''), str) for k in ('path', 'text', 'profile', 'direction')):
                raise ValueError()
            if type(row.get('duration')) not in (int, float) or not 0 < row['duration'] <= 1200:
                raise ValueError()
        for row in data.get('pending', []):
            if not ident(row.get('id')) or not isinstance(row.get('server'), str) or not isinstance(row.get('request'), dict):
                raise ValueError()
            if row['request'].get('engine') not in ENGINES:
                raise ValueError()

    def save(self):
        os.makedirs(self.directory, exist_ok=True)
        temp = self.path + '.tmp'
        with open(temp, 'w', encoding='utf8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, self.path)

    def add_take(self, audio, request, *, ident=None, profile='', server='', stats=None):
        stats = dict(stats, **check_wav(audio, analyze=False)) if stats is not None else check_wav(audio)
        ident = ident or uuid.uuid4().hex
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise VoiceError('Invalid local take ID.')
        os.makedirs(self.directory, exist_ok=True)
        target = os.path.join(self.directory, ident + '.wav')
        if os.path.exists(target):
            # A previous WAV write may have succeeded before library.json
            # failed. Recover that exact take without overwriting other audio.
            if os.path.getsize(target) != len(audio):
                raise VoiceError('This local take ID already contains different audio.')
            with open(target, 'rb') as f:
                if f.read(MAX_AUDIO + 1) != audio:
                    raise VoiceError('This local take ID already contains different audio.')
        else:
            with open(target, 'xb') as f:
                f.write(audio)
        existing = next((row for row in self.data['takes'] if row['id'] == ident), None)
        if existing:
            self.save()
            return existing
        record = dict(id=ident, path=target, engine=request['engine'], text=request['text'],
                      direction=request.get('direction', ''), seed=request['seed'], language=request['language'],
                      profile=profile, server=server, created=time.time(), sha256=hashlib.sha256(audio).hexdigest(), **stats)
        self.data['takes'].insert(0, record)
        self.save()
        return record

    def export(self, take, destination):
        source = os.path.join(self.directory, take['id'] + '.wav')
        if os.path.abspath(destination) == os.path.abspath(source):
            return
        shutil.copyfile(source, destination)
