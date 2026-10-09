"""Screen source-detail-preserving coordinate fallbacks without retraining."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from ml.experiments.phase6g_2_identity_margin.train import (
    MultiScaleCorrespondenceFaceSwapModel6G1, CelebAPairedDataset,
    RealFacePreprocessor, prepare_validation_pairs,
)
from ml.inference.face_correspondence import preprocess_single_face
from ml.training.robust_correspondence import visibility_warp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).parent/'run/best_model.pt')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'visibility_screen')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    model = MultiScaleCorrespondenceFaceSwapModel6G1(base_channels=64, bottleneck_channels=512,
        embedding_dim=512, fullres_channels=32, blur_kernel_size=9, blur_sigma=3,
        arcface_checkpoint_path=str(ROOT/'ml/models/weights/ms1mv2_iresnet50.pth')).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict']); del state
    model.eval()
    pre = RealFacePreprocessor(image_size=128)
    val = CelebAPairedDataset(dataset_root=str(ROOT/'ml/data/celeba'), split='val', seed=42)
    pairs = prepare_validation_pairs(val, pre, args.pairs, args.seed)
    assert len(pairs) == args.pairs
    def u8(t):
        return ((t.squeeze(0).cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    records, rows = [], []
    variants = ['legacy', 'visibility_0.25', 'visibility_0.5', 'visibility_1.0']
    with torch.inference_mode():
        for index, pair in enumerate(pairs):
            si, _, _, _, dense, _ = preprocess_single_face(pre, str(val.img_dir_path/pair['src_name']))
            src, tgt = pair['source_tensor'].to(device), pair['target_tensor'].to(device)
            zs, zt = model.arcface(src), model.arcface(tgt)
            record = dict(source=pair['src_name'], target=pair['tgt_name'])
            tiles = [u8(src),u8(tgt)]
            for name in variants:
                if name == 'legacy':
                    aligned, conf = pair['aligned_source'], pair['confidence_map']
                else:
                    aligned, conf, _ = visibility_warp(si,dense,pair['target_dense'],strength=float(name.split('_')[1]))
                result = model(i_source=src,i_target=tgt,l_target=pair['target_l_map'].to(device),
                    aligned_source=aligned.to(device),confidence_map=conf.to(device))
                comp = result['i_composite']; rgb = u8(comp)
                zc = model.arcface(comp)
                lm = None
                try:
                    detection = pre.detect_landmarks(Image.fromarray(rgb))
                    lm = float(np.linalg.norm(detection.key_landmarks_5pts-pair['target_5pts'],axis=1).mean())
                except Exception:
                    pass
                record[name] = dict(A=float(F.cosine_similarity(zs,zc)),B=float(F.cosine_similarity(zt,zc)),
                    landmark_error=lm,sharpness=float(cv2.Laplacian(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()))
                tiles.append(rgb)
            records.append(record)
            if index < 24: rows.append(tiles)
            if (index+1)%25 == 0: print(f'Evaluated {index+1}/{len(pairs)}',flush=True)
    aggregate = {}
    for name in variants:
        subset = [record[name] for record in records]
        aggregate[name] = dict(A=float(np.mean([r['A'] for r in subset])),
            B=float(np.mean([r['B'] for r in subset])),
            source_wins=100*float(np.mean([r['A']>r['B'] for r in subset])),
            redetection=100*float(np.mean([r['landmark_error'] is not None for r in subset])),
            landmark_error=float(np.mean([r['landmark_error'] for r in subset if r['landmark_error'] is not None])),
            sharpness=float(np.mean([r['sharpness'] for r in subset])))
    (args.output/'results.json').write_text(json.dumps(dict(seed=args.seed,checkpoint=str(args.checkpoint),aggregate=aggregate,pairs=records),indent=2))
    for group in range((len(rows)+5)//6):
        batch = rows[group*6:(group+1)*6]
        canvas = Image.new('RGB',(768,len(batch)*150),'white'); draw=ImageDraw.Draw(canvas)
        for row, tiles in enumerate(batch):
            for col,(tile,label) in enumerate(zip(tiles,['SOURCE','TARGET',*variants])):
                canvas.paste(Image.fromarray(tile),(col*128,row*150+22))
                draw.text((col*128+2,row*150+3),label,fill='black')
        canvas.save(args.output/f'comparison_{group+1:02d}.png')
    print(json.dumps(aggregate),flush=True)


if __name__ == '__main__': main()
