import unittest
from ml.detection.call_audio import call_audio_result, normalize_call_pcm
import numpy as np

class CallAudioTest(unittest.TestCase):
    def test_decoder_overshoot_is_accepted_and_bounded(self):
        pcm = np.array([1.03, -.4, -1.02],dtype=np.float32)
        result = normalize_call_pcm(pcm)
        np.testing.assert_allclose(result,[1,-.4,-1])
        self.assertGreater(pcm[0],1)

    def test_invalid_pcm_is_rejected(self):
        for value in [[float('nan')],[float('inf')],[5],[[.1,.2]]]:
            with self.assertRaises(ValueError):normalize_call_pcm(value)

    def test_unvalidated_output_is_not_a_spoof_measurement(self):
        source = dict(status='measured', score=.999, model='AASIST')
        result = call_audio_result(source)
        self.assertIsNone(result['score'])
        self.assertEqual(result['raw_score'], .999)
        self.assertEqual(result['status'], 'unvalidated')
        self.assertEqual(source['score'], .999)

    def test_silence_remains_insufficient(self):
        source = dict(status='insufficient', score=None, reason='No speech')
        self.assertEqual(call_audio_result(source), source)

if __name__ == '__main__': unittest.main()
