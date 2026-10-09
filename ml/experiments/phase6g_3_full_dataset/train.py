"""Resumable full-training-split CelebA fine-tuning (identity-disjoint validation)."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from ml.experiments.phase6g_2_identity_margin.train import (
    MultiScaleCorrespondenceFaceSwapModel6G1, Stage2CompositeLoss,
    PatchGANDiscriminator, AdversarialLoss, CelebAPairedDataset,
    RealFacePreprocessor, prepare_validation_pairs, run_validation,
    differentiable_embedding, identity_margin, EXPECTED_ARCFACE_SHA,
)
from ml.inference.face_correspondence import preprocess_single_face, warp_and_prepare_source
from ml.training.face_preprocessing import NoFaceDetectedError
from ml.training.robust_correspondence import robust_warp, reliable_legacy_warp, visibility_warp
from ml.training.phase6d_landmark_correspondence import compute_pose_metrics


class CurriculumRejected(Exception):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def identity_embedding_fp32(self, image):
    # ArcFace input gradients can overflow fp16; keep the frozen identity
    # network in fp32 while the generator/discriminator use mixed precision.
    with torch.amp.autocast('cuda', enabled=False):
        image = F.interpolate(image.float(), size=(112, 112), mode='bilinear', align_corners=False)
        return self.backbone(image)


def prepare_pair(pre, source, target, correspondence='legacy', curriculum_limits=None):
    si, st, _, _, sd, s5 = preprocess_single_face(pre, str(source))
    _, tt, mask, landmarks, td, t5 = preprocess_single_face(pre, str(target))
    if curriculum_limits is not None:
        sp, tp = compute_pose_metrics(s5), compute_pose_metrics(t5)
        disparity = np.hypot(sp['yaw_ratio']-tp['yaw_ratio'], sp['pitch_ratio']-tp['pitch_ratio'])
        if disparity > curriculum_limits[0]:
            raise CurriculumRejected('pose_disparity')
    if correspondence in ('robust', 'reliable', 'visibility'):
        warp = {'robust': robust_warp, 'reliable': reliable_legacy_warp,
                'visibility': lambda *a: visibility_warp(*a, strength=0.25)}[correspondence]
        aligned, confidence, diagnostics = warp(si, sd, td)
        if curriculum_limits is not None and diagnostics['unreliable_pixel_fraction'] > curriculum_limits[1]:
            raise CurriculumRejected('unreliable_geometry')
    else:
        aligned, confidence = warp_and_prepare_source(si, sd, td, 128)
    return dict(source=torch.from_numpy(st).unsqueeze(0), target=torch.from_numpy(tt).unsqueeze(0),
        target_mask=torch.from_numpy(mask).reshape(1, 1, 128, 128),
        target_landmark_map=torch.from_numpy(landmarks).unsqueeze(0),
        aligned_source=aligned, confidence_map=confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--val-pairs', type=int, default=100)
    parser.add_argument('--val-interval', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=0, help='0 processes the complete epoch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--quality-tuning', action='store_true', help='Stronger self reconstruction, edge supervision and opaque face core')
    parser.add_argument('--same-identity-views', action='store_true', help='Use different photos of the same identity for genuine target pixel supervision')
    parser.add_argument('--correspondence', choices=['legacy', 'robust', 'reliable', 'visibility'], default='legacy')
    parser.add_argument('--pose-curriculum', action='store_true')
    parser.add_argument('--initialization', type=Path, default=ROOT / 'ml/experiments/phase6g_2_identity_margin/checkpoints/best_model.pt')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'run')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.val_pairs, args.val_interval) < 1 or args.max_steps < 0:
        parser.error('Counts must be positive; max-steps may be zero.')
    if args.pose_curriculum and (args.correspondence not in ('reliable', 'visibility') or not args.max_steps):
        parser.error('Pose curriculum requires reliable or visibility correspondence and a bounded max-steps schedule.')
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'training.pid').write_text(str(os.getpid()))
    latest = outdir / 'latest_model.pt'
    if (outdir / 'status.json').exists() and not args.resume:
        raise RuntimeError('Use --resume to continue existing output, or a new --output directory.')
    if args.resume and not latest.exists():
        raise RuntimeError('Resume checkpoint missing.')
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required.')
    device = torch.device('cuda')
    arc = ROOT / 'ml/models/weights/ms1mv2_iresnet50.pth'
    assert hashlib.sha256(arc.read_bytes()).hexdigest().upper() == EXPECTED_ARCFACE_SHA
    initialization = args.initialization.resolve()
    model = MultiScaleCorrespondenceFaceSwapModel6G1(base_channels=64, bottleneck_channels=512,
        embedding_dim=512, fullres_channels=32, blur_kernel_size=9, blur_sigma=3,
        arcface_checkpoint_path=str(arc)).to(device)
    state = torch.load(latest if args.resume else initialization, map_location=device, weights_only=False)
    if args.resume:
        for key in ('epochs', 'batch_size', 'val_pairs', 'val_interval', 'correspondence', 'quality_tuning', 'pose_curriculum', 'max_steps', 'same_identity_views'):
            default = 'legacy' if key == 'correspondence' else False if key in ('quality_tuning', 'pose_curriculum', 'same_identity_views') else None
            if state['config'].get(key, default) != getattr(args, key):
                raise ValueError(f'Resume requires matching {key}.')
    model.load_state_dict(state['model_state_dict'])
    model.arcface.forward = types.MethodType(identity_embedding_fp32, model.arcface)
    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)
    discriminator.load_state_dict(state['discriminator_state_dict'])
    criterion = Stage2CompositeLoss(model.arcface, w_id=10, w_id_swap=8,
        w_struct=7 if args.quality_tuning else 5, w_bg=5, w_mask=5, w_adv=0.5).to(device)
    adversarial = AdversarialLoss().to(device)
    opt_g = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    sg, sd = torch.amp.GradScaler('cuda', enabled=False), torch.amp.GradScaler('cuda', enabled=False)
    pre = RealFacePreprocessor(image_size=128)
    dataset_root = str(ROOT / 'ml/data/celeba')
    train = CelebAPairedDataset(dataset_root=dataset_root, split='train', seed=42)
    val = CelebAPairedDataset(dataset_root=dataset_root, split='val', seed=42)
    assert set(train.split_identity_to_images).isdisjoint(val.split_identity_to_images)
    ids = sorted(train.split_identity_to_images)
    samples = sorted(train.samples, key=lambda item: item[1].name)
    random.Random(43).shuffle(samples)
    pairs = prepare_validation_pairs(val, pre, args.val_pairs, seed=137)
    if len(pairs) != args.val_pairs:
        raise RuntimeError('Incomplete validation sample.')
    if args.correspondence in ('robust', 'reliable', 'visibility'):
        for pair in pairs:
            si, _, _, _, source_dense, _ = preprocess_single_face(pre, str(train.img_dir_path / pair['src_name']))
            warp = {'robust': robust_warp, 'reliable': reliable_legacy_warp,
                    'visibility': lambda *a: visibility_warp(*a, strength=0.25)}[args.correspondence]
            pair['aligned_source'], pair['confidence_map'], _ = warp(si, source_dense, pair['target_dense'])
    baseline = state['baseline'] if args.resume else run_validation(model, pairs, pre, device)
    best = state['best'] if args.resume else baseline
    cursor, step, skipped = (state['cursor'], state['step'], state['skipped']) if args.resume else (0, 0, 0)
    rng = random.Random(44)
    if args.resume:
        for key, optimizer in [('optimizer_g', opt_g), ('optimizer_d', opt_d)]:
            optimizer.load_state_dict(state[key])
        sg.load_state_dict(state['scaler_g'])
        sd.load_state_dict(state['scaler_d'])
        rng.setstate(state['pair_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    total = len(samples) * args.epochs
    started = time.time()
    config = dict(epochs=args.epochs, batch_size=args.batch_size, training_images=len(samples),
        validation_images=len(val), total_dataset_images=len(train)+len(val), training_identities=len(ids),
        validation_identities=len(val.split_identity_to_images), identity_overlap=0,
        validation_seed=137, initialization=str(initialization), lr=1e-5,
        self_reconstruction_fraction=0.4 if args.quality_tuning else 0.2, margin=0.2, margin_weight=10,
        reconstruction_weight=12 if args.quality_tuning else 5,
        quality_tuning=args.quality_tuning, self_edge_weight=3 if args.quality_tuning else 0,
        core_opacity_weight=3 if args.quality_tuning else 0,
        val_interval=args.val_interval, val_pairs=args.val_pairs, max_steps=args.max_steps,
        correspondence=args.correspondence)
    config['pose_curriculum'] = args.pose_curriculum
    config['same_identity_views'] = args.same_identity_views
    config['curriculum_schedule'] = [[0.15, 0.10], [0.35, 0.20], [None, None]] if args.pose_curriculum else None
    curriculum_rejected = state.get('curriculum_rejected', 0) if args.resume else 0
    atomic_json(outdir / 'config.json', config)
    print(json.dumps(config), flush=True)

    def save(name):
        temporary = outdir / (name + '.tmp')
        torch.save(dict(model_state_dict=model.state_dict(), discriminator_state_dict=discriminator.state_dict(),
            optimizer_g=opt_g.state_dict(), optimizer_d=opt_d.state_dict(), scaler_g=sg.state_dict(), scaler_d=sd.state_dict(),
            cursor=cursor, step=step, skipped=skipped, curriculum_rejected=curriculum_rejected, baseline=baseline, best=best,
            pair_rng=rng.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), config=config), temporary)
        temporary.replace(outdir / name)

    def status(completed=False, metrics=None):
        atomic_json(outdir / 'status.json', dict(step=step, target_images_processed=cursor,
            total_target_images=total, skipped_pairs=skipped, completed=completed,
            elapsed_seconds=time.time()-started, curriculum_rejected=curriculum_rejected,
            curriculum_stage=min(2, step*3//args.max_steps) if args.pose_curriculum else None,
            baseline=baseline, best=best, latest_metrics=metrics))

    if not args.resume:
        save('best_model.pt')
        save('latest_model.pt')
    status()
    batch, self_flags = [], []
    while cursor < total and (not args.max_steps or step < args.max_steps):
        identity, target = samples[cursor % len(samples)]
        cursor += 1
        self_pair = rng.random() < config['self_reconstruction_fraction']
        if self_pair:
            source = target
            if args.same_identity_views:
                choices = [image for image in train.split_identity_to_images[identity] if image != target]
                if choices:
                    source = rng.choice(choices)
        else:
            other = rng.choice(ids)
            while other == identity:
                other = rng.choice(ids)
            source = rng.choice(train.split_identity_to_images[other])
        try:
            stage = min(2, step*3//args.max_steps) if args.pose_curriculum else 2
            limits = [(0.15,0.10),(0.35,0.20),None][stage] if args.pose_curriculum else None
            batch.append(prepare_pair(pre, source, target, args.correspondence, limits))
            self_flags.append(self_pair)
        except CurriculumRejected:
            curriculum_rejected += 1
        except (NoFaceDetectedError, AttributeError, ValueError, RuntimeError, OSError) as error:
            skipped += 1
            with (outdir / 'skipped.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(source=source.name, target=target.name, error=str(error)))+'\n')
            if skipped > max(100, cursor * 0.1):
                raise RuntimeError('Preprocessing failure rate exceeded 10%.')
        if len(batch) < args.batch_size and cursor < total:
            continue
        if not batch:
            continue
        tensors = {k: torch.cat([p[k] for p in batch]).to(device) for k in batch[0]}
        model.train()
        discriminator.train()
        for p in discriminator.parameters(): p.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            result = model(i_source=tensors['source'], i_target=tensors['target'],
                l_target=tensors['target_landmark_map'], aligned_source=tensors['aligned_source'], confidence_map=tensors['confidence_map'])
            loss, _ = criterion(i_source=tensors['source'], i_target=tensors['target'],
                i_swap=result['i_swap'], i_composite=result['i_composite'], pred_mask=result['m_pred'],
                target_mask=tensors['target_mask'], d_fake_logits=discriminator(result['i_composite']))
            cross = torch.tensor([not flag for flag in self_flags], device=device)
            if cross.any():
                loss = loss + 10 * identity_margin(model.arcface, tensors['source'][cross], tensors['target'][cross],
                    result['i_composite'][cross], result['i_swap'][cross], 0.2)
            same = ~cross
            # Same identity provides genuine target supervision, including
            # different-photo pairs when requested. No cross-ID pixel target.
            if same.any():
                mask = tensors['target_mask'][same]
                reconstruction = ((result['i_swap'][same]-tensors['target'][same]).abs()*mask).sum() / (mask.sum()*3+1e-6)
                loss = loss + config['reconstruction_weight'] * reconstruction
                if args.quality_tuning:
                    # Exact self-pair edges provide real supervision without
                    # forcing another identity's eyes or mouth onto cross-ID outputs.
                    prediction, reference = result['i_swap'][same], tensors['target'][same]
                    dx = ((prediction[:,:,:,1:]-prediction[:,:,:,:-1])-(reference[:,:,:,1:]-reference[:,:,:,:-1])).abs()
                    dy = ((prediction[:,:,1:,:]-prediction[:,:,:-1,:])-(reference[:,:,1:,:]-reference[:,:,:-1,:])).abs()
                    loss = loss + 3 * ((dx*mask[:,:,:,1:]).sum()+(dy*mask[:,:,1:,:]).sum())/(6*mask.sum()+1e-6)
            if args.quality_tuning:
                # Erode supervision to avoid making the feathered boundary opaque.
                core = 1-F.max_pool2d(1-tensors['target_mask'], kernel_size=9, stride=1, padding=4)
                loss = loss + 3 * ((1-result['m_pred'])*core).sum()/(core.sum()+1e-6)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite generator loss.')
        sg.scale(loss).backward()
        sg.unscale_(opt_g)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5, error_if_nonfinite=True)
        sg.step(opt_g)
        sg.update()
        for p in discriminator.parameters(): p.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            ld, _, _ = adversarial.discriminator_loss(discriminator(tensors['target']), discriminator(result['i_composite'].detach()))
        if not torch.isfinite(ld): raise RuntimeError('Nonfinite discriminator loss.')
        sd.scale(ld).backward()
        sd.step(opt_d)
        sd.update()
        step += 1
        batch, self_flags = [], []
        if step % 25 == 0:
            status()
            print(f'Step {step}; targets {cursor}/{total}; loss {loss.item():.4f}; grad {float(norm):.3f}; skipped {skipped}', flush=True)
        if step % 250 == 0: save('latest_model.pt')
        if step % args.val_interval == 0 or cursor == total or step == args.max_steps:
            metrics = run_validation(model, pairs, pre, device)
            metrics['step'] = step
            eligible = (metrics['A'] >= baseline['A']*0.95 and metrics['sharp_comp'] >= baseline['sharp_comp']*0.9
                and metrics['redetect_rate'] >= baseline['redetect_rate']-1
                and metrics['mean_lm_err'] is not None and metrics['mean_lm_err'] <= baseline['mean_lm_err']+0.5)
            if eligible and (metrics['pct_A_gt_B'], metrics['A_minus_B']) > (best['pct_A_gt_B'], best['A_minus_B']):
                best = metrics
                save('best_model.pt')
            with (outdir / 'validation.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
            # Comparable sample strips accompany quantitative validation.
            model.eval()
            with torch.no_grad():
                strips = []
                for pair in pairs[:6]:
                    predicted = model(i_source=pair['source_tensor'].to(device), i_target=pair['target_tensor'].to(device),
                        l_target=pair['target_l_map'].to(device), aligned_source=pair['aligned_source'].to(device), confidence_map=pair['confidence_map'].to(device))
                    tiles = [pair['source_tensor'], pair['target_tensor'], pair['aligned_source'], predicted['i_composite']]
                    strips.append(np.concatenate([((t.squeeze(0).cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8) for t in tiles], axis=1))
                Image.fromarray(np.concatenate(strips, axis=0)).save(outdir / f'validation_{step:06d}.png')
            save('latest_model.pt')
            status(metrics=metrics)
            print('VALIDATION '+json.dumps(metrics), flush=True)
    save('latest_model.pt')
    status(completed=cursor >= total)
    print('Full epoch complete.' if cursor >= total else 'Configured step limit reached.', flush=True)


if __name__ == '__main__':
    main()
