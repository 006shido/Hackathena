# Full-dataset CelebA training

Started 2026-10-05 from the selected Phase 6G.2 checkpoint. One epoch traverses all 172,111 training images as targets in shuffled order. The remaining 30,488 images (1,527 identities) are held out; the training split contains 8,650 identities. Cross-identity sources are sampled dynamically from the training split. This is full-training-split coverage, not a 300-pair pool. Some images may fail landmark preprocessing; failures are recorded and excessive failure rates halt the run.

20% of pairs are self-reconstruction examples with exact pixel supervision. Cross-identity examples retain source identity and margin supervision alongside the existing structure, background, mask and adversarial objectives. There is no ground-truth cross-identity swapped image in CelebA. This run is face-generator training, not audio or video detector training.

Precision: bfloat16 generator/discriminator, full precision frozen ArcFace with image gradients enabled. fp16 overflow was caught during preflight; the bfloat16 five-step test and six-step checkpoint-resume test completed successfully. GPU: RTX 5060 Laptop, 8 GB.

## Progress and artifacts

- `training.log`: preprocessing, progress every 25 steps, validation results.
- `training.log` also includes library diagnostics or exceptions (MediaPipe warnings alone are not training failures).
- `run/training.pid`: Python PID. The current run uses persistent command session 87759; the initial detached launch did not persist.
- `run/config.json`: exact dataset counts and configuration.
- `run/status.json`: latest cursor, training step, failures, validation and completion state.
- `run/latest_model.pt`: resumable model, discriminator, optimizers and random states; saved every 250 steps.
- `run/best_model.pt`: selected candidate, initially the starting model. Promotion requires identity improvement with quality checks.
- `run/validation.jsonl`: metrics every 1,000 steps, plus final evaluation.
- `run/validation_*.png`: fixed comparison images at each evaluation.
- `run/skipped.jsonl`: failed preprocessing pairs, if any.

The epoch needs roughly 43,000 optimizer steps; failed pairs change this count. Full-image coverage is recorded by the target cursor, not inferred from step count. Dataset and prior checkpoints are preserved. Best-checkpoint retention protects against degraded later weights; successful completion does not imply photorealistic swaps. Visual review and fresh evaluation are still needed afterwards.

## Resume after interruption

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'C:/files/forgitclone/Hackathena/.venv/Lib/site-packages'
& 'C:/Users/shido/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -u ml/experiments/phase6g_3_full_dataset/train.py --resume
```

Run this only after the previous training process has stopped. A resume continues from the last saved checkpoint, including source-pair RNG and GPU RNG state. It may repeat work since that checkpoint. `status.json` can be ahead of the durable checkpoint. The training uses the bundled Python because the existing virtual environment's Microsoft Store launcher is unavailable.

Keep the computer awake, plugged in, and Codex open while training. The run is in a persistent command session; there is no scheduled monitor or automatic notification configured.
