"""Fresh validation sample, comparative visuals, and model-only latency."""
import sys
import json
import time
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from PIL import Image, ImageDraw
from ml.experiments.phase6g_2_identity_margin.train import (
    MultiScaleCorrespondenceFaceSwapModel6G1, CelebAPairedDataset,
    RealFacePreprocessor, prepare_validation_pairs, run_validation, NumpyEncoder,
)


def main():
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--candidate-correspondence', choices=['legacy', 'reliable', 'visibility'], default='legacy')
    args = parser.parse_args()
    output = args.output or directory / 'evaluation'
    output.mkdir(exist_ok=True)
    device = torch.device('cuda')
    model = MultiScaleCorrespondenceFaceSwapModel6G1(base_channels=64, bottleneck_channels=512,
        embedding_dim=512, fullres_channels=32, blur_kernel_size=9, blur_sigma=3,
        arcface_checkpoint_path=str(ROOT / 'ml/models/weights/ms1mv2_iresnet50.pth')).to(device)
    pre = RealFacePreprocessor(image_size=128)
    val = CelebAPairedDataset(dataset_root=str(ROOT / 'ml/data/celeba'), split='val', seed=42)
    pairs = prepare_validation_pairs(val, pre, num_pairs=300, seed=args.seed)
    assert len(pairs) == 300
    results = dict(seed=args.seed, count=len(pairs),
        limitation='Fresh sample from the validation identity split, not a separate test split.',
        pairs=[dict(source=p['src_name'], target=p['tgt_name']) for p in pairs], metrics={})
    visuals = {}
    checkpoints = [('baseline', ROOT / 'ml/experiments/phase6g_2_identity_margin/checkpoints/best_model.pt'),
            ('selected', directory / 'run/best_model.pt'), ('final', args.candidate or directory / 'run/latest_model.pt')]
    for name, checkpoint in checkpoints:
        if name == 'final' and args.candidate_correspondence in ('reliable', 'visibility'):
            from ml.training.robust_correspondence import reliable_legacy_warp, visibility_warp
            from ml.inference.face_correspondence import preprocess_single_face
            for pair in pairs:
                si, _, _, _, dense, _ = preprocess_single_face(pre, str(val.img_dir_path / pair['src_name']))
                warp = reliable_legacy_warp if args.candidate_correspondence == 'reliable' else lambda *a: visibility_warp(*a,strength=0.25)
                pair['aligned_source'], pair['confidence_map'], _ = warp(si, dense, pair['target_dense'])
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        del state
        model.eval()
        metrics = run_validation(model, pairs, pre, device)
        results['metrics'][name] = metrics
        print(name + ' ' + json.dumps(metrics), flush=True)
        def forward(pair):
            return model(i_source=pair['source_tensor'].to(device), i_target=pair['target_tensor'].to(device),
                l_target=pair['target_l_map'].to(device), aligned_source=pair['aligned_source'].to(device), confidence_map=pair['confidence_map'].to(device))
        with torch.no_grad():
            visuals[name] = [forward(p)['i_composite'].cpu() for p in pairs[:24]]
            for _ in range(5): forward(pairs[0])
            latency = []
            for p in pairs[:30]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                forward(p)
                torch.cuda.synchronize()
                latency.append(1000*(time.perf_counter()-start))
            metrics['forward_ms_median'] = float(np.median(latency))
            metrics['forward_ms_p95'] = float(np.percentile(latency, 95))
        (output / 'results.json').write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    def pixels(t):
        return ((t.squeeze(0).permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    for group in range(4):
        canvas = Image.new('RGB', (768, 6*150), 'white')
        draw = ImageDraw.Draw(canvas)
        for row in range(6):
            index = group*6+row
            pair = pairs[index]
            tiles = [pair['source_tensor'], pair['target_tensor'], pair['aligned_source'],
                visuals['baseline'][index], visuals['selected'][index], visuals['final'][index]]
            for col, (tile, label) in enumerate(zip(tiles, ['SOURCE','TARGET','ALIGNED','BASELINE','SELECTED','FINAL'])):
                canvas.paste(Image.fromarray(pixels(tile)), (col*128,row*150+22))
                draw.text((col*128+3,row*150+3), label, fill='black')
        canvas.save(output / f'comparison_{group+1:02d}.png')
    print('Fresh evaluation complete.', flush=True)


if __name__ == '__main__':
    main()
