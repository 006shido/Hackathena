"""Separate unvalidated call-audio logits from usable anomaly measurements."""
import numpy as np

def normalize_call_pcm(pcm):
    pcm = np.asarray(pcm, dtype=np.float32)
    # Decoded Opus can overshoot full scale even when encoder input was clipped.
    # Accept bounded decoder headroom, then match model training's input range.
    if pcm.ndim != 1 or not np.isfinite(pcm).all() or np.max(np.abs(pcm), initial=0) > 4:
        raise ValueError('Expected finite mono PCM with bounded decoder headroom.')
    return np.clip(pcm, -1, 1)

def call_audio_result(result):
    output = dict(result)
    if output.get('status') == 'measured':
        output['raw_score'] = output.pop('score', None)
        output.update(status='unvalidated', score=None,
                      reason='Reliable call-audio spoof score unavailable: this model also flags clean call speech.')
    return output
