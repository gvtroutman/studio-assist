"""Remote-only stdlib job server. ML runs in configured, contained children.

Run on the LLM PC, never inside the workstation's Studio Assist process.
Each job has a caller-generated ID: an uncertain POST can be inspected rather
than repeated. Cancelling waits for the owned process to exit before settling.
"""
import argparse
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core import procs
from .catalog import ENGINES, MAX_AUDIO, MAX_REFERENCE, PROTOCOL, VoiceError, validate
from .client import check_wav

ID = re.compile('[0-9a-f]{32}\\Z')
MAX_BODY = MAX_REFERENCE * 4 // 3 + 128 * 1024


class Jobs:
    def __init__(self, directory, environments, runner=None):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.environments = environments
        self.records = {}
        self.lock = threading.RLock()
        self.pending = queue.Queue(maxsize=6)
        self.stopping = threading.Event()
        self.runner = runner or self.run_worker
        self.restore()
        self.thread = threading.Thread(target=self.work, name='VoiceQueue', daemon=True)
        self.thread.start()

    def restore(self):
        """Retain completed output and fail interrupted jobs without replay."""
        for folder in self.root.iterdir():
            if not ID.fullmatch(folder.name) or not folder.is_dir() or len(self.records) >= 256:
                continue
            try:
                status_path = folder / 'status.json'
                if status_path.stat().st_size > 4096:
                    continue
                saved = json.loads(status_path.read_text(encoding='utf8'))
                if saved.get('id') != folder.name or saved.get('engine') not in ENGINES:
                    continue
                if saved.get('status') not in ('completed', 'failed', 'cancelled'):
                    saved.update(status='failed', error='Server restarted during this job. No generation was replayed.', progress='Interrupted')
                saved.update(cancel=threading.Event(), child=None, folder=folder)
                self.records[folder.name] = saved
                self.persist(folder.name)
            except (OSError, ValueError, TypeError):
                continue

    def capabilities(self):
        engines = []
        for ident, spec in ENGINES.items():
            config = self.environments.get(spec['group'], {})
            python = config.get('python', '')
            engines.append(dict(id=ident, name=spec['name'], configured=bool(python and os.path.isfile(python)),
                                license=spec['license'], reference=spec['reference'],
                                direction=spec['direction'], cpu=spec.get('cpu', False)))
        return dict(protocol=PROTOCOL, engines=engines, serial=True,
                    note='Configured environments load cached models, or download on first use when allowed.')

    def submit(self, data):
        ident = data.get('id') if isinstance(data, dict) else None
        if not isinstance(ident, str) or not ID.fullmatch(ident):
            raise VoiceError('Supply a 32-character hexadecimal job ID.')
        raw = data.get('reference')
        reference = None
        if raw is not None:
            if not isinstance(raw, str) or len(raw) > MAX_REFERENCE * 4 // 3 + 4:
                raise VoiceError('Reference audio exceeds the size limit.')
            try:
                reference = base64.b64decode(raw, validate=True)
            except ValueError:
                raise VoiceError('Invalid reference audio encoding.') from None
            if len(reference) > MAX_REFERENCE:
                raise VoiceError('Reference audio exceeds the size limit.')
            metadata = check_wav(reference, max_seconds=30)
            if data.get('engine') == 'chatterbox' and metadata['duration'] <= 5:
                raise VoiceError('Chatterbox Turbo needs a reference longer than five seconds.')
        request = validate(data, reference=bool(reference))
        spec = ENGINES[request['engine']]
        config = self.environments.get(spec['group'], {})
        if not config.get('python') or not os.path.isfile(config['python']):
            raise VoiceError('%s has no configured environment on this server. Run remote setup first.' % spec['name'])
        canonical = {k: request.get(k, default) for k, default in (
            ('engine', ''), ('text', ''), ('language', 'en'), ('seed', 0), ('direction', ''),
            ('reference_text', ''), ('speaker', ''), ('allow_download', False))}
        fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode() + (reference or b'')).hexdigest()
        with self.lock:
            existing = self.records.get(ident)
            if existing:
                if existing.get('fingerprint') != fingerprint:
                    raise VoiceError('This job ID already belongs to a different request.')
                return self.status(ident)
            # Bound retained jobs. Files are deliberately retained for recovery.
            if len(self.records) >= 256:
                raise VoiceError('This server has retained 256 jobs. Archive its data and restart with a new data folder.')
            if self.pending.full():
                raise VoiceError('The voice queue is full. Wait for an active job to finish.')
            folder = self.root / ident
            if folder.exists():
                raise VoiceError('This job exists on disk from an earlier server session. Inspect it before retrying with a new ID.')
            folder.mkdir()
            if reference:
                (folder / 'reference.wav').write_bytes(reference)
            canonical.update(id=ident, reference_path=str(folder / 'reference.wav') if reference else '',
                             config=config)
            (folder / 'request.json').write_text(json.dumps(canonical), encoding='utf8')
            self.records[ident] = dict(id=ident, engine=request['engine'], status='queued', progress='Waiting for the GPU',
                                      fingerprint=fingerprint, cancel=threading.Event(), child=None, folder=folder)
            self.pending.put_nowait(ident)
            self.persist(ident)
            return self.status(ident)

    def status(self, ident):
        with self.lock:
            row = self.records.get(ident)
            if row is None:
                raise KeyError(ident)
            return {k: row[k] for k in ('id', 'engine', 'status', 'progress', 'error') if k in row}

    def persist(self, ident):
        row = self.records[ident]
        target = row['folder'] / 'status.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(self.status(ident), fingerprint=row.get('fingerprint', ''))), encoding='utf8')
        temp.replace(target)

    def cancel(self, ident):
        with self.lock:
            row = self.records.get(ident)
            if row is None:
                raise KeyError(ident)
            if row['status'] not in ('completed', 'failed', 'cancelled'):
                row['cancel'].set()
                if row['status'] == 'queued':
                    row.update(status='cancelled', error='Voice generation cancelled.', progress='Cancelled before generation')
                else:
                    row['status'] = 'cancelling'
                    row['progress'] = 'Stopping the owned worker'
                self.persist(ident)
            return self.status(ident)

    def work(self):
        while not self.stopping.is_set():
            try:
                ident = self.pending.get(timeout=.2)
            except queue.Empty:
                continue
            row = self.records[ident]
            try:
                if not row['cancel'].is_set():
                    with self.lock:
                        row.update(status='running', progress='Loading model and generating speech')
                        self.persist(ident)
                    self.runner(row)
                with self.lock:
                    if row['cancel'].is_set():
                        row.update(status='cancelled', error='Voice generation cancelled.', progress='Cancelled')
                    else:
                        audio_path = row['folder'] / 'audio.wav'
                        if audio_path.stat().st_size > MAX_AUDIO:
                            raise VoiceError('Generated audio exceeds 64 MB.')
                        check_wav(audio_path.read_bytes())
                        row.update(status='completed', progress='Complete; worker released')
            except Exception as error:
                with self.lock:
                    row.update(status='cancelled' if row['cancel'].is_set() else 'failed',
                               error=str(error)[:600], progress='Stopped')
            finally:
                with self.lock:
                    row['child'] = None
                    try:
                        self.persist(ident)
                    except OSError:
                        traceback.print_exc()  # Keep processing other jobs; status remains inspectable in memory.
                self.pending.task_done()

    def run_worker(self, row):
        folder = row['folder']
        request = json.loads((folder / 'request.json').read_text(encoding='utf8'))
        config = request['config']
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        if config.get('source'):
            env['PYTHONPATH'] = os.pathsep.join((config['source'], str(Path(config['source']) / 'third_party/Matcha-TTS'), str(repo_root)))
        else:
            env['PYTHONPATH'] = str(repo_root)
        env['HF_HUB_OFFLINE'] = '0' if request['allow_download'] else '1'
        env['TRANSFORMERS_OFFLINE'] = env['HF_HUB_OFFLINE']
        env['HF_HUB_DISABLE_TELEMETRY'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'
        args = [config['python'], '-m', 'apps.voice_studio.worker', str(folder / 'request.json')]
        log_path = folder / 'worker.log'
        with log_path.open('wb') as log:
            child = procs.spawn(args, cwd=str(repo_root), env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=procs.NO_WINDOW)
            with self.lock:
                row['child'] = child
            deadline = time.monotonic() + 1800
            try:
                while child.proc.poll() is None:
                    if row['cancel'].is_set() or self.stopping.is_set():
                        child.kill()
                        break
                    if time.monotonic() > deadline or log_path.stat().st_size > 4 * 1024 * 1024:
                        raise VoiceError('The remote worker exceeded its time or log limit. See ' + str(log_path))
                    progress_file = folder / 'progress.json'
                    if progress_file.exists():
                        try:
                            progress = json.loads(progress_file.read_text(encoding='utf8'))
                            with self.lock:
                                row['progress'] = str(progress.get('progress', 'Generating'))[:200]
                        except (ValueError, OSError):
                            pass
                    row['cancel'].wait(.15)
                if child.proc.poll() is None:
                    raise VoiceError('Worker termination was not confirmed; inspect the remote server.')
                if child.proc.returncode and not row['cancel'].is_set():
                    error_path = folder / 'error.json'
                    message = 'Speech worker failed. See ' + str(log_path)
                    if error_path.exists():
                        message = json.loads(error_path.read_text(encoding='utf8')).get('error', message)
                    raise VoiceError(message)
            finally:
                child.kill()

    def audio(self, ident):
        if self.status(ident)['status'] != 'completed':
            raise VoiceError('This job has no completed audio.')
        return (self.records[ident]['folder'] / 'audio.wav').read_bytes()

    def close(self):
        self.stopping.set()
        with self.lock:
            for row in self.records.values():
                row['cancel'].set()
        self.thread.join(timeout=8)


def make_server(address, jobs, token=''):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Requests contain no useful diagnostics; worker logs are per job.

        def send(self, status, value, audio=False):
            data = value if audio else json.dumps(value).encode('utf8')
            self.send_response(status)
            self.send_header('Content-Type', 'audio/wav' if audio else 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def dispatch(self, post=False):
            if token and not hmac.compare_digest(self.headers.get('Authorization', '').encode('utf8'), ('Bearer ' + token).encode('utf8')):
                self.send(401, {'error': 'Server token required.'})
                return
            self.connection.settimeout(15)
            route = self.path.split('/')
            try:
                body = None
                if post:
                    size = int(self.headers.get('Content-Length', '-1'))
                    if not 0 <= size <= MAX_BODY:
                        raise VoiceError('Request body exceeds the limit.')
                    body = json.loads(self.rfile.read(size))
                if self.path == '/v1/capabilities' and not post:
                    self.send(200, jobs.capabilities())
                elif self.path == '/v1/jobs' and post:
                    self.send(202, jobs.submit(body))
                elif len(route) in (4, 5) and route[1:3] == ['v1', 'jobs'] and ID.fullmatch(route[3]):
                    ident = route[3]
                    if len(route) == 4 and not post:
                        self.send(200, jobs.status(ident))
                    elif len(route) == 5 and route[4] == 'audio' and not post:
                        self.send(200, jobs.audio(ident), audio=True)
                    elif len(route) == 5 and route[4] == 'cancel' and post:
                        self.send(200, jobs.cancel(ident))
                    else:
                        self.send(404, {'error': 'Unknown voice endpoint.'})
                else:
                    self.send(404, {'error': 'Unknown voice endpoint.'})
            except KeyError:
                self.send(404, {'error': 'Unknown job. Retain the local job ID and inspect remote recovery files.'})
            except (VoiceError, ValueError, TypeError) as error:
                self.send(400, {'error': str(error)[:600]})
            except (OSError, TimeoutError):
                self.send(500, {'error': 'The voice server could not complete the request; inspect its data folder.'})

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(True)

    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description='Studio Assist voice server — run on the LLM PC')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=17494)
    parser.add_argument('--config', default='voice-environments.json')
    parser.add_argument('--data-dir', default='voice-jobs')
    args = parser.parse_args()
    token = os.environ.get('STUDIO_VOICE_TOKEN', '')
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == 'localhost'
    if not loopback and len(token) < 24:
        parser.error('Set STUDIO_VOICE_TOKEN to a random token of at least 24 characters for a remote listener.')
    with open(args.config, encoding='utf8') as f:
        environments = json.load(f)
    jobs = Jobs(args.data_dir, environments)
    server = make_server((args.host, args.port), jobs, token)
    print('Voice server listening on %s:%d; one isolated worker at a time.' % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        jobs.close()
        procs.stop_all(0)


if __name__ == '__main__':
    main()
