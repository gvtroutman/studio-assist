"""Real voice protocol and persistence, with synthetic PCM instead of models."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid
import wave
import zipfile

from apps.voice_studio import catalog, client, remote


def wav(seconds=.1, value=1234):
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(8000)
        target.writeframes(int(value).to_bytes(2, 'little', signed=True) * round(8000 * seconds))
    return stream.getvalue()


def request(**values):
    return dict(dict(engine='voxcpm', text='The same spoken line.', seed=123, language='en',
                     direction='Warm narration', reference_text='', allow_download=False), **values)


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.started = []
        self.active = 0
        self.max_active = 0
        self.gate = None
        self.entered = threading.Event()
        def runner(row):
            self.started.append(row['id'])
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entered.set()
            try:
                if self.gate:
                    while not self.gate.is_set() and not row['cancel'].wait(.01):
                        pass
                (row['folder'] / 'audio.wav').write_bytes(wav(value=1111))
            finally:
                self.active -= 1
        self.environments = {'voxcpm': {'python': sys.executable, 'device': 'cpu'}}
        self.jobs = remote.Jobs(self.temp.name, self.environments, runner=runner)
        self.addCleanup(self.jobs.close)
        self.server = remote.make_server(('127.0.0.1', 0), self.jobs, token='test-token')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = client.Client('http://127.0.0.1:%d' % self.server.server_port, 'test-token')

    def wait(self, ident, terminal=True):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.jobs.status(ident)
            if (result['status'] in ('completed', 'failed', 'cancelled')) == terminal:
                return result
            time.sleep(.01)
        self.fail('Job did not settle: ' + str(result))

    def test_actual_protocol_returns_valid_audio_and_isolated_capabilities(self):
        caps = self.client.capabilities()
        self.assertTrue(caps['serial'])
        self.assertEqual({e['id'] for e in caps['engines'] if e['configured']}, {'voxcpm'})
        audio, result = self.client.generate(request())
        self.assertEqual(audio, wav(value=1111))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(client.check_wav(audio)['sample_rate'], 8000)
        self.assertEqual(len(self.started), 1)

    def test_same_id_never_generates_twice_and_mismatch_is_refused(self):
        ident = uuid.uuid4().hex
        self.client.request('/jobs', dict(request(), id=ident))
        self.wait(ident)
        self.client.request('/jobs', dict(request(), id=ident))
        self.assertEqual(self.started, [ident])
        with self.assertRaisesRegex(catalog.VoiceError, 'different request'):
            self.client.request('/jobs', dict(request(text='Different line'), id=ident))

    def test_queue_serializes_and_cancelled_queued_job_never_runs(self):
        self.gate = threading.Event()
        one, two, three = [uuid.uuid4().hex for _ in range(3)]
        self.client.request('/jobs', dict(request(), id=one))
        self.assertTrue(self.entered.wait(2))
        self.client.request('/jobs', dict(request(), id=two))
        self.client.request('/jobs', dict(request(), id=three))
        self.assertEqual(self.client.request('/jobs/' + two + '/cancel', {})['status'], 'cancelled')
        self.assertNotIn(two, self.started)
        self.gate.set()
        self.wait(three)
        self.assertEqual(self.max_active, 1)
        self.assertEqual(self.started, [one, three])

    def test_running_cancellation_and_recovery_do_not_post_again(self):
        self.gate = threading.Event()
        ident = uuid.uuid4().hex
        self.client.request('/jobs', dict(request(), id=ident))
        self.assertTrue(self.entered.wait(2))
        self.client.request('/jobs/' + ident + '/cancel', {})
        self.assertEqual(self.wait(ident)['status'], 'cancelled')
        self.assertEqual(self.active, 0)
        with self.assertRaisesRegex(catalog.VoiceError, 'cancelled'):
            self.client.collect(ident, request())
        self.assertEqual(self.started, [ident])

    def test_completed_job_survives_restart_without_replay(self):
        ident = uuid.uuid4().hex
        self.client.request('/jobs', dict(request(), id=ident))
        self.wait(ident)
        self.jobs.close()
        restarted = remote.Jobs(self.temp.name, self.environments, runner=lambda _r: self.fail('Replayed'))
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.status(ident)['status'], 'completed')
        self.assertEqual(restarted.audio(ident), wav(value=1111))
        restarted.submit(dict(request(), id=ident))
        self.assertEqual(restarted.pending.qsize(), 0)

    def test_missing_auth_bad_audio_and_unconfigured_engine_refused(self):
        with self.assertRaisesRegex(catalog.VoiceError, 'refused access'):
            client.Client(self.client.url).capabilities()
        with self.assertRaisesRegex(catalog.VoiceError, 'no configured environment'):
            self.client.request('/jobs', dict(request(engine='kokoro'), id=uuid.uuid4().hex))
        with self.assertRaises(catalog.VoiceError):
            self.client.generate(request(), reference=b'not a wav')
        self.assertFalse(self.started)

    def test_server_rejects_invalid_reference_and_path_ids(self):
        with self.assertRaisesRegex(catalog.VoiceError, 'Invalid reference'):
            self.client.request('/jobs', dict(request(), id=uuid.uuid4().hex, reference='!'))
        with self.assertRaises(catalog.VoiceError):
            self.client.request('/jobs/../../config')
        self.assertFalse(self.started)


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_stats_and_export_preserve_exact_wav_and_request_identity(self):
        library = client.Library(self.temp.name)
        record = library.add_take(wav(value=32767), request(), profile='Narrator', server='http://voice')
        self.assertEqual(record['clipped_samples'], 800)
        self.assertEqual(record['seed'], 123)
        exported = os.path.join(self.temp.name, 'exported.wav')
        library.export(record, exported)
        self.assertEqual(Path(exported).read_bytes(), wav(value=32767))
        self.assertEqual(client.Library(self.temp.name).data['takes'][0]['id'], record['id'])
        self.assertNotIn('token', json.dumps(library.data))

    def test_take_recovers_after_metadata_save_failed_without_overwriting_audio(self):
        library = client.Library(self.temp.name)
        ident = uuid.uuid4().hex
        with patch.object(library, 'save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                library.add_take(wav(), request(), ident=ident)
        recovered = client.Library(self.temp.name)
        recovered.add_take(wav(), request(), ident=ident)
        self.assertEqual(len(client.Library(self.temp.name).data['takes']), 1)
        with self.assertRaisesRegex(catalog.VoiceError, 'different audio'):
            recovered.add_take(wav(value=2), request(), ident=ident)

    def test_corrupt_library_is_preserved_and_not_overwritten(self):
        target = Path(self.temp.name, 'library.json')
        target.write_text('broken', encoding='utf8')
        with self.assertRaisesRegex(catalog.VoiceError, 'recovery'):
            client.Library(self.temp.name)
        self.assertEqual(target.read_text(encoding='utf8'), 'broken')

    def test_malformed_nested_library_records_are_preserved(self):
        target = Path(self.temp.name, 'library.json')
        for data in ({'profiles': [None]}, {'draft': []}, {'pending': [{'id': '../unsafe'}]},
                     {'takes': [{'id': uuid.uuid4().hex, 'engine': 'voxcpm', 'duration': 'oops'}]}):
            original = json.dumps(data)
            target.write_text(original, encoding='utf8')
            with self.assertRaises(catalog.VoiceError):
                client.Library(self.temp.name)
            self.assertEqual(target.read_text(encoding='utf8'), original)

    def test_truncated_audio_and_unsafe_local_id_refused(self):
        with self.assertRaisesRegex(catalog.VoiceError, 'truncated'):
            client.check_wav(wav()[:-4])
        with self.assertRaises(catalog.VoiceError):
            client.Library(self.temp.name).add_take(wav(), request(), ident='../outside')

    def test_chunking_does_not_drop_unspaced_or_punctuated_words(self):
        for script in ('One sentence. Two more! ' * 100, '字' * 1200, 'A' * 1900):
            parts = catalog.chunks(script)
            self.assertEqual(''.join(''.join(parts).split()), ''.join(script.split()))
            self.assertTrue(all(0 < len(p) <= 450 for p in parts))

    def test_engine_capabilities_refuse_false_promises(self):
        with self.assertRaisesRegex(catalog.VoiceError, 'needs a reference'):
            catalog.validate(request(engine='qwen-clone'))
        with self.assertRaisesRegex(catalog.VoiceError, 'words spoken'):
            catalog.validate(request(engine='omnivoice'), reference=True)
        with self.assertRaisesRegex(catalog.VoiceError, 'does not support'):
            catalog.validate(request(engine='chatterbox', language='fr'), reference=True)
        with self.assertRaises(catalog.VoiceError):
            catalog.validate(request(seed=True))

    def test_setup_bundle_contains_only_remote_source_and_verifiable_manifest(self):
        from apps.voice_studio.bundle import export
        import hashlib
        output = export(Path(self.temp.name, 'remote.zip'))
        with zipfile.ZipFile(output) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            self.assertNotIn('apps/voice_studio/ui.py', archive.namelist())
            self.assertIn('core/procs.py', manifest)
            for member, digest in manifest.items():
                self.assertEqual(hashlib.sha256(archive.read(member)).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
