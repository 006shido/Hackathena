"""Continue the best 6G.1 weights with explicit source-over-target supervision.

Outputs are isolated; existing checkpoints and the live application are preserved.
"""
import argparse
import hashlib
import json
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from ml.experiments.phase6g_1_fullres_skip.train_phase6g1 import (
    EXPECTED_ARCFACE_SHA, prepare_training_pool, prepare_validation_pairs,
    run_validation, NumpyEncoder,
)
from ml.experiments.phase6g_1_fullres_skip.model_phase6g1 import MultiScaleCorrespondenceFaceSwapModel6G1
from ml.models.losses import Stage2CompositeLoss
from ml.models.discriminator import PatchGANDiscriminator, AdversarialLoss
from ml.data.dataset import CelebAPairedDataset
from ml.training.face_preprocessing import RealFacePreprocessor


def differentiable_embedding(self, image):
    """Freeze weights, not the gradient through generated pixels.

    The legacy extractor wraps the backbone in no_grad(), disconnecting all
    identity losses. Override only this experimental instance.
    """
    image = F.interpolate(image, size=(112, 112), mode='bilinear', align_corners=False)
    return self.backbone(image)


def identity_margin(arcface, source, target, composite, swap, margin):
    # Reference embeddings are frozen; output embeddings retain image gradients.
    with torch.no_grad():
        zs, zt = arcface(source), arcface(target)
    losses = []
    for output in (composite, swap):
        zo = arcface(output)
        a = F.cosine_similarity(zs.float(), zo.float())
        b = F.cosine_similarity(zt.float(), zo.float())
        losses.append(F.relu(b - a + margin).mean())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--pool-size', type=int, default=300)
    parser.add_argument('--val-pairs', type=int, default=100)
    parser.add_argument('--val-interval', type=int, default=25)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--margin-weight', type=float, default=10.0)
    args = parser.parse_args()
    if min(args.steps, args.val_interval, args.val_pairs) < 1 or args.pool_size < 4:
        parser.error('Positive step/validation counts and a pool of at least four are required.')
    if args.margin < 0 or args.margin_weight < 0:
        parser.error('Margin and weight must be nonnegative.')
    torch.manual_seed(42)
    random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this controlled training run.')
    device = torch.device('cuda')
    output = Path(__file__).parent / 'checkpoints'
    output.mkdir(exist_ok=True)
    if (output / 'metrics.json').exists():
        raise RuntimeError('Existing results found; preserve them before starting another run.')
    arc = ROOT / 'ml/models/weights/ms1mv2_iresnet50.pth'
    assert hashlib.sha256(arc.read_bytes()).hexdigest().upper() == EXPECTED_ARCFACE_SHA
    init = ROOT / 'ml/experiments/phase6g_1_fullres_skip/checkpoints/best_model.pt'
    protected = [arc, init, ROOT / 'ml/checkpoints/stage2/phase6g/best_model.pt']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    model = MultiScaleCorrespondenceFaceSwapModel6G1(
        base_channels=64, bottleneck_channels=512, embedding_dim=512,
        fullres_channels=32, blur_kernel_size=9, blur_sigma=3.0,
        arcface_checkpoint_path=str(arc)).to(device)
    state = torch.load(init, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.arcface.forward = types.MethodType(differentiable_embedding, model.arcface)
    probe = torch.randn(1, 3, 128, 128, device=device, requires_grad=True)
    identity_margin(model.arcface, probe.detach(), -probe.detach(), probe, probe, 2.0).backward()
    assert probe.grad is not None and torch.isfinite(probe.grad).all() and probe.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in model.arcface.parameters())
    print('Identity loss gradient verified; ArcFace weights remain frozen.', flush=True)
    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)
    discriminator.load_state_dict(state['discriminator_state_dict'])
    criterion = Stage2CompositeLoss(model.arcface, w_id=10, w_id_swap=8,
        w_struct=5, w_bg=5, w_mask=5, w_adv=0.5).to(device)
    adversarial = AdversarialLoss().to(device)
    opt_g = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-5, betas=(0.5, 0.999))
    scaler_g, scaler_d = torch.amp.GradScaler('cuda'), torch.amp.GradScaler('cuda')
    pre = RealFacePreprocessor(image_size=128)
    dataset_root = str(ROOT / 'ml/data/celeba')
    train = CelebAPairedDataset(dataset_root=dataset_root, split='train', seed=42)
    val = CelebAPairedDataset(dataset_root=dataset_root, split='val', seed=42)
    assert set(train.split_identity_to_images).isdisjoint(val.split_identity_to_images)
    pairs = prepare_validation_pairs(val, pre, args.val_pairs, seed=42)
    pool = prepare_training_pool(train, pre, args.pool_size, seed=123)
    if len(pairs) != args.val_pairs or len(pool) < 4:
        raise RuntimeError('Insufficient verified pairs; refusing incomplete evaluation.')
    baseline = run_validation(model, pairs, pre, device)
    baseline['step'] = 0
    history = [baseline]
    best = baseline

    def save(name, metrics):
        torch.save(dict(model_state_dict=model.state_dict(),
            discriminator_state_dict=discriminator.state_dict(),
            optimizer_g=opt_g.state_dict(), optimizer_d=opt_d.state_dict(),
            scaler_g=scaler_g.state_dict(), scaler_d=scaler_d.state_dict(),
            step=metrics['step'], metrics=metrics, config=vars(args)), output / name)

    save('best_model.pt', baseline)
    rng = random.Random(42)
    for step in range(1, args.steps + 1):
        model.train()
        discriminator.train()
        batch = rng.sample(pool, 4)
        tensors = {key: torch.cat([p[key] for p in batch]).to(device) for key in batch[0]}
        for p in discriminator.parameters():
            p.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(i_target=tensors['target'], l_target=tensors['target_landmark_map'],
                i_source=tensors['source'], aligned_source=tensors['aligned_source'],
                confidence_map=tensors['confidence_map'])
            loss, _ = criterion(i_source=tensors['source'], i_target=tensors['target'],
                i_swap=out['i_swap'], i_composite=out['i_composite'], pred_mask=out['m_pred'],
                target_mask=tensors['target_mask'], d_fake_logits=discriminator(out['i_composite']))
            margin_loss = identity_margin(model.arcface, tensors['source'], tensors['target'],
                out['i_composite'], out['i_swap'], args.margin)
            loss = loss + args.margin_weight * margin_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite generator loss at step {step}')
        scaler_g.scale(loss).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
        scaler_g.step(opt_g)
        scaler_g.update()
        for p in discriminator.parameters():
            p.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss_d, _, _ = adversarial.discriminator_loss(discriminator(tensors['target']), discriminator(out['i_composite'].detach()))
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()
        if step % args.val_interval == 0 or step == args.steps:
            metrics = run_validation(model, pairs, pre, device)
            metrics.update(step=step, margin_loss=float(margin_loss.detach()))
            history.append(metrics)
            # Reject identity wins accompanied by material quality regressions.
            eligible = (metrics['A'] >= baseline['A'] * 0.95
                and metrics['sharp_comp'] >= baseline['sharp_comp'] * 0.9
                and metrics['redetect_rate'] >= baseline['redetect_rate']
                and metrics['mean_lm_err'] is not None
                and metrics['mean_lm_err'] <= baseline['mean_lm_err'] + 0.5)
            rank = lambda m: (m['pct_A_gt_B'], m['A_minus_B'])
            if eligible and rank(metrics) > rank(best):
                best = metrics
                save('best_model.pt', metrics)
            save('latest_model.pt', metrics)
            report = dict(config=vars(args), initialization=str(init), baseline=baseline,
                best=best, history=history, completed=step == args.steps,
                classification='IMPROVED' if best['pct_A_gt_B'] > baseline['pct_A_gt_B'] else 'NEEDS TUNING',
                protected_hashes=hashes)
            (output / 'metrics.json').write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
            print(f"Step {step}: A={metrics['A']:.4f} B={metrics['B']:.4f} source wins={metrics['pct_A_gt_B']:.1f}% sharpness={metrics['sharp_comp']:.1f}", flush=True)
    for p in protected:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
    model.load_state_dict(torch.load(output / 'best_model.pt', map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for index, pair in enumerate(pairs[:8]):
            result = model(i_source=pair['source_tensor'].to(device),
                i_target=pair['target_tensor'].to(device), l_target=pair['target_l_map'].to(device),
                aligned_source=pair['aligned_source'].to(device), confidence_map=pair['confidence_map'].to(device))
            tiles = [pair['source_tensor'], pair['target_tensor'], pair['aligned_source'], result['i_swap'], result['i_composite']]
            pixels = [((t.detach().squeeze(0).cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8) for t in tiles]
            Image.fromarray(np.concatenate(pixels, axis=1)).save(output / f'comparison_{index + 1:02d}.png')


if __name__ == '__main__':
    main()
