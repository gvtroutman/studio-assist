"""Audio is uploaded to a fake host only. No creative app or model is started."""
import base64
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import wave
from unittest.mock import patch

from core import audio_review as ar


def wav(path, seconds=2):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b'\x12\x01' * (8000 * seconds))


def review():
    return {'audio_assessed':True,'summary':'T2 has a cleaner ending.','best_take':'T2',
            'ranking':['T2','T1'],'takes':[
                {'id':'T1','strengths':['Clear words'],'concerns':['Abrupt ending'],
                 'confidence':.8,'evidence':[{'start':1,'end':1.5,'note':'Ending cuts off.'}]},
                {'id':'T2','strengths':['Smooth ending'],'concerns':[],
                 'confidence':.9,'evidence':[{'start':1,'end':1.5,'note':'Ending is intact.'}]}]}


class AudioTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths=[os.path.join(self.tmp.name,'first.wav'),os.path.join(self.tmp.name,'second.wav')]
        for p in self.paths: wav(p)
        self.takes=[{'path':p} for p in self.paths]
        self.requests=[]

    def post(self, req, timeout):
        self.requests.append(req)
        self.assertEqual(timeout,ar.TIMEOUT)
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':json.dumps(review())}}]}).encode())

    def call(self, **kwargs):
        return ar.compare(self.takes,'http://audio-host:8000/v1','audio-model',post=self.post,**kwargs)

    def test_actual_audio_sent_and_timestamps_retained(self):
        result=self.call(script='Hello.',api_key='secret')
        body=json.loads(self.requests[0].data)
        self.assertEqual(body['model'],'audio-model')
        self.assertEqual(body['modalities'],['text'])
        audio=[p['input_audio'] for p in body['messages'][1]['content'] if p['type']=='input_audio']
        self.assertEqual(len(audio),2)
        data=base64.b64decode(audio[0]['data'])
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual(w.readframes(1),b'\x12\x01')
        self.assertEqual(result['best_take'],'T2')
        self.assertEqual(result['sources'][0]['duration'],2)
        self.assertNotIn('secret',json.dumps(result))
        self.assertEqual(self.requests[0].get_header('Authorization'),'Bearer secret')
        self.assertIn('second.wav',ar.report_text(result))

    def test_exact_source_range_preserves_samples(self):
        data,meta=ar.prepare_take({'path':self.paths[0],'start':.5,'end':1.25})
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual(w.getnframes(),6000)
            self.assertEqual(w.readframes(1),b'\x12\x01')
        self.assertEqual(meta['source_start'],.5)
        self.assertEqual(meta['duration'],.75)

    def test_ranges_and_size_rejected_before_network(self):
        for take in [{'path':self.paths[0],'start':3}, {'path':self.paths[0],'end':3},
                     {'path':self.paths[0],'start':1,'end':.5}, {'path':self.paths[0],'start':'nan'},
                     {'path':'key.pem'}]:
            with self.subTest(take=take), self.assertRaises(ar.AudioReviewError):
                ar.prepare_take(take)
        with patch.object(ar.os.path,'getsize',return_value=ar.MAX_FILE_BYTES+1):
            with self.assertRaises(ar.AudioReviewError): ar.prepare_take(self.takes[0])
        self.assertFalse(self.requests)

    def test_long_recording_needs_range(self):
        wav(self.paths[0],61)
        with self.assertRaisesRegex(ar.AudioReviewError,'exceeds 60'):
            ar.prepare_take(self.takes[0])
        self.assertEqual(ar.prepare_take({'path':self.paths[0],'start':60,'end':61})[1]['duration'],1)

    def test_unsupported_audio_does_not_retry_or_fallback(self):
        def reject(req,timeout):
            self.requests.append(req)
            raise urllib.error.HTTPError(req.full_url,400,'unsupported input_audio',{},None)
        with self.assertRaisesRegex(ar.AudioReviewError,'rejected audio input'):
            ar.compare(self.takes,'http://host','text-model',post=reject)
        self.assertEqual(len(self.requests),1)

    def test_cancellation_before_and_after_network(self):
        event=threading.Event();event.set()
        with self.assertRaisesRegex(ar.AudioReviewError,'cancelled'): self.call(cancel=event)
        self.assertFalse(self.requests)
        event.clear()
        def post(req,timeout):
            event.set()
            return self.post(req,timeout)
        with self.assertRaisesRegex(ar.AudioReviewError,'cancelled'):
            ar.compare(self.takes,'http://host','audio-model',cancel=event,post=post)

    def test_invalid_model_reply_cannot_select_winner(self):
        meta=[{'id':'T1','duration':2},{'id':'T2','duration':2}]
        bad=[]
        r=review();r['audio_assessed']=False;bad.append(r)
        r=review();r['ranking']=['T1','T1'];bad.append(r)
        r=review();r['best_take']='T1';bad.append(r)
        r=review();r['takes'][0]['evidence'][0]['end']=4;bad.append(r)
        r=review();r['takes'][1]['evidence']=[];bad.append(r)
        r=review();r['takes'][0]['confidence']=float('nan');bad.append(r)
        for r in bad:
            with self.subTest(r=r),self.assertRaises(ar.AudioReviewError):
                ar.clean_result(json.dumps(r),meta)
        with self.assertRaises(ar.AudioReviewError): ar.clean_result('T2 is good',meta)

    def test_tie_is_preserved(self):
        r=review();r['best_take']=None
        self.assertIsNone(ar.clean_result(json.dumps(r),[{'id':'T1','duration':2},{'id':'T2','duration':2}])['best_take'])

    def test_endpoint_and_credential_boundaries(self):
        self.assertEqual(ar.endpoint('http://host:8000'),'http://host:8000/v1/chat/completions')
        self.assertEqual(ar.endpoint('http://host/v1/'),'http://host/v1/chat/completions')
        self.assertEqual(ar.endpoint('https://host/api/v1/chat/completions'),'https://host/api/v1/chat/completions')
        for url in ('file:///secret','https://key@host','https://host?api_key=secret'):
            with self.assertRaises(ar.AudioReviewError): ar.endpoint(url)
        redirect=ar._NoRedirect()
        self.assertIsNone(redirect.redirect_request(None,None,302,'',{},'https://other-host'))

    def test_total_upload_limit(self):
        with patch.object(ar,'MAX_AUDIO_BYTES',100):
            with self.assertRaises(ar.AudioReviewError): self.call()
        self.assertFalse(self.requests)

    def test_unconfigured_host_or_model_does_not_send(self):
        for host,model in [('', 'audio-model'),('http://host','')]:
            with self.assertRaises(ar.AudioReviewError): ar.compare(self.takes,host,model,post=self.post)
        self.assertFalse(self.requests)


class AudioWindowTest(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from core.audio_review_ui import AudioReviewWindow
        from core.ui import DARK
        try: self.root=tk.Tk();self.root.withdraw()
        except tk.TclError: self.skipTest('No display')
        self.saved=[]
        self.win=AudioReviewWindow(self.root,palette=DARK,save_settings=self.saved.append)
        self.addCleanup(self.root.destroy)
        self.addCleanup(self.win.close)

    def test_validation_and_partial_settings(self):
        self.win.start_review()
        self.assertFalse(self.win.running)
        self.assertFalse(self.saved)
        self.assertIn('host',self.win.status.get())

    def test_primary_controls_fit_window(self):
        self.root.deiconify()  # transient windows stay unmapped while the parent is withdrawn
        self.win.deiconify()
        self.root.update()
        button=self.win.compare_button
        self.assertLessEqual(button.winfo_rooty()+button.winfo_height(),
                             self.win.winfo_rooty()+self.win.winfo_height())
        self.assertGreaterEqual(self.win.output.winfo_height(),60)

    def test_stale_result_ignored_and_timer_cancelled(self):
        self.win.cancel_event.set()
        self.win.running=True
        self.win.events.put(('result',{'unexpected':'must not render'}))
        self.win.events.put(('done',None))
        self.win.drain()
        self.assertIsNone(self.win.result)
        self.assertIsNone(self.win.timer)
        self.assertFalse(self.win.running)

    def test_closed_window_discards_queued_result(self):
        self.win.close()
        self.win.events.put(('result',{'unexpected':'must not render'}))
        self.win.drain()
        self.assertTrue(self.win.closed)
        self.assertTrue(self.win.cancel_event.is_set())

    def test_compare_uses_worker_and_never_persists_key(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            paths=[os.path.join(tmp,'one.wav'),os.path.join(tmp,'two.wav')]
            for p in paths: wav(p)
            self.win.takes=[{'path':p,'start':'0','end':''} for p in paths]
            self.win.host.set('http://host:8000/v1')
            self.win.model.set('audio-model')
            self.win.key.set('must-not-save')
            metadata=[{'id':'T1','label':'one','path':paths[0],'source_start':0,'duration':2},
                      {'id':'T2','label':'two','path':paths[1],'source_start':0,'duration':2}]
            result=review();result.update(model='audio-model',sources=metadata,basis='Direct audio')
            with patch.object(ar,'compare',return_value=result) as compare:
                self.win.start_review()
                deadline=time.monotonic()+2
                while self.win.running and time.monotonic()<deadline:
                    self.root.update()
                    time.sleep(.01)
                self.assertFalse(self.win.running)
                self.assertEqual(self.win.result['best_take'],'T2')
                self.assertEqual(compare.call_args.kwargs['api_key'],'must-not-save')
                self.assertEqual(self.saved,[{'host':'http://host:8000/v1','model':'audio-model'}])
                self.assertNotIn('must-not-save',self.win.output.get('1.0','end'))


if __name__=='__main__': unittest.main()
