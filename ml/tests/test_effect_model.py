import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np,torch
from ml.detection.effect_model import CallEffectModel,spectral_statistics
from ml.detection.call_features import call_features

class EffectModelTest(unittest.TestCase):
    def make_candidate(self,path,passed=True):
        torch.save(dict(mean=torch.zeros(64),std=torch.ones(64),weight=torch.zeros(1,64),bias=torch.zeros(1),threshold=.8),path/'effect_candidate.pt')
        (path/'effect_results.json').write_text(json.dumps(dict(passed_local_gate=passed,sha256=hashlib.sha256((path/'effect_candidate.pt').read_bytes()).hexdigest())))

    def test_training_and_inference_feature_parity(self):
        rng=np.random.default_rng(3);pcm=rng.normal(0,.1,64600).astype(np.float32)
        expected=call_features(torch.zeros(1,160),torch.zeros(1,2),pcm)[-64:]
        np.testing.assert_allclose(spectral_statistics(pcm),expected,rtol=1e-6)

    def test_failed_candidate_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'detection/call_domain') as name:
            path=Path(name);self.make_candidate(path,False)
            with self.assertRaisesRegex(ValueError,'failed local validation'):CallEffectModel(path)

    def test_changed_checkpoint_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'detection/call_domain') as name:
            path=Path(name);self.make_candidate(path)
            with (path/'effect_candidate.pt').open('ab') as output:output.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):CallEffectModel(path)

    def test_prediction_and_gain_invariance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'detection/call_domain') as name:
            path=Path(name);self.make_candidate(path);model=CallEffectModel(path)
            pcm=(.1*np.sin(2*np.pi*170*np.arange(64600)/16000)).astype(np.float32)
            self.assertEqual(model.score(pcm),.5)
            np.testing.assert_allclose(spectral_statistics(pcm),spectral_statistics(pcm*.5),atol=1e-4)

if __name__=='__main__':unittest.main()
