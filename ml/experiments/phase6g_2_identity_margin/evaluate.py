"""Fresh-seed comparison and visual outputs for the selected checkpoint."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from PIL import Image
from ml.experiments.phase6g_2_identity_margin.train import (
    MultiScaleCorrespondenceFaceSwapModel6G1, CelebAPairedDataset,
    RealFacePreprocessor, prepare_validation_pairs, run_validation, NumpyEncoder,
)

def main():
    output = Path(__file__).parent / 'checkpoints'
    device = torch.device('cuda')
    model = MultiScaleCorrespondenceFaceSwapModel6G1(base_channels=64,
        bottleneck_channels=512, embedding_dim=512, fullres_channels=32,
        blur_kernel_size=9, blur_sigma=3.0,
        arcface_checkpoint_path=str(ROOT / 'ml/models/weights/ms1mv2_iresnet50.pth')).to(device)
    pre = RealFacePreprocessor(image_size=128)
    val = CelebAPairedDataset(dataset_root=str(ROOT / 'ml/data/celeba'), split='val', seed=42)
    pairs = prepare_validation_pairs(val, pre, num_pairs=100, seed=137)
    assert len(pairs) == 100
    results = {'seed': 137, 'pairs': [dict(source=p['src_name'], target=p['tgt_name']) for p in pairs]}
    for name, checkpoint in [('baseline', ROOT / 'ml/experiments/phase6g_1_fullres_skip/checkpoints/best_model.pt'), ('candidate', output / 'best_model.pt')]:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        results[name] = run_validation(model, pairs, pre, device)
        print(name, results[name], flush=True)
    model.eval()
    with torch.no_grad():
        for index, pair in enumerate(pairs[:12]):
            result = model(i_source=pair['source_tensor'].to(device),
                i_target=pair['target_tensor'].to(device), l_target=pair['target_l_map'].to(device),
                aligned_source=pair['aligned_source'].to(device), confidence_map=pair['confidence_map'].to(device))
            tiles = [pair['source_tensor'], pair['target_tensor'], pair['aligned_source'], result['i_swap'], result['i_composite']]
            pixels = [((t.detach().squeeze(0).cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8) for t in tiles]
            Image.fromarray(np.concatenate(pixels, axis=1)).save(output / f'fresh_comparison_{index + 1:02d}.png')
    (output / 'fresh_evaluation.json').write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == '__main__':
    main()
