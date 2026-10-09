"""
FastAPI Localhost ML Service for Phase 6G Face Swapping
======================================================
Loads Phase 6G Neural Pipeline (Phase6GInferenceEngine) as a singleton at startup.
Exposes:
  - GET  /health: Verifies GPU, model checkpoints, and engine readiness
  - POST /infer:  Accepts source and target face images, executes inference, and
                  returns base64-encoded output images and complete diagnostic metrics.
"""

import os
import sys
import time
import uuid
import base64
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.inference.infer_phase6g import Phase6GInferenceEngine, EXPECTED_ARCFACE_SHA
from ml.training.face_preprocessing import FaceDetectionError, NoFaceDetectedError
from ml.api.video_preview import router as video_preview_router, run_worker
from ml.api.media_detection import router as media_detection_router
from ml.api.webrtc_video import router as webrtc_video_router, close_all as close_video_peers

# Maximum allowed file upload size: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_data_url(path):
    if not path or not Path(path).exists():
        return ""
    return f"data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode('utf-8')}"

CHECKPOINT_PATH = Path(os.environ.get('ML_CHECKPOINT_PATH', str(REPO_ROOT / "ml/checkpoints/stage2/phase6g/best_model.pt")))
if not CHECKPOINT_PATH.is_absolute():
    CHECKPOINT_PATH = REPO_ROOT / CHECKPOINT_PATH
MODEL_VARIANT = os.environ.get('ML_MODEL_VARIANT', 'phase6g')
CORRESPONDENCE = os.environ.get('ML_CORRESPONDENCE', 'legacy')
ARCFACE_PATH = REPO_ROOT / "ml/models/weights/ms1mv2_iresnet50.pth"
MEDIAPIPE_TASK_PATH = REPO_ROOT / "client/public/models/face_landmarker.task"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & shutdown lifecycle management.
    Loads Phase 6G model ONCE at startup onto GPU.
    """
    print("[Phase 6G API] Initializing service startup checks...", flush=True)

    # 1. Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Phase 6G requires an NVIDIA GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"[Phase 6G API] Detected GPU: {gpu_name}", flush=True)

    # 2. Verify Checkpoints and Assets
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Phase 6G checkpoint not found at: {CHECKPOINT_PATH}")

    if not ARCFACE_PATH.exists():
        raise FileNotFoundError(f"ArcFace checkpoint not found at: {ARCFACE_PATH}")

    if not MEDIAPIPE_TASK_PATH.exists():
        raise FileNotFoundError(f"MediaPipe face landmarker asset not found at: {MEDIAPIPE_TASK_PATH}")

    print("[Phase 6G API] Checkpoint & asset verification PASSED.", flush=True)

    # 3. Instantiate Singleton Inference Engine
    print("[Phase 6G API] Loading Phase 6G model into GPU memory...", flush=True)
    t_load_start = time.time()
    engine = Phase6GInferenceEngine(
        checkpoint_path=str(CHECKPOINT_PATH),
        arcface_path=str(ARCFACE_PATH),
        device=torch.device("cuda:0"),
        model_variant=MODEL_VARIANT,
        correspondence=CORRESPONDENCE
    )
    load_time_sec = time.time() - t_load_start
    print(f"[Phase 6G API] Phase 6G loaded successfully in {load_time_sec:.2f}s.", flush=True)

    # 4. Engine Warmup (JIT compilation, cuDNN, MediaPipe landmarker initialization)
    dummy_s = REPO_ROOT / "ml/data/celeba/img_align_celeba/000001.jpg"
    dummy_t = REPO_ROOT / "ml/data/celeba/img_align_celeba/000002.jpg"
    if not dummy_s.exists() or not dummy_t.exists():
        # Fresh clones use the existing demo portrait; CelebA is not redistributed.
        dummy_s = dummy_t = REPO_ROOT / 'client/public/astronaut.jpg'
    if dummy_s.exists() and dummy_t.exists():
        print("[Phase 6G API] Running engine warmup...", flush=True)
        t_warmup = time.time()
        with tempfile.TemporaryDirectory() as td:
            engine.infer(str(dummy_s), str(dummy_t), output_dir=td, pair_prefix="warmup")
        print(f"[Phase 6G API] Engine warmup complete in {time.time() - t_warmup:.2f}s.", flush=True)

    if os.environ.get('ML_ENABLE_VIDEO_PREVIEW')=='1' and os.environ.get('ML_VIDEO_BACKEND')=='research-reference':
        from ml.experiments.reference_backend.opencv_reference import OpenCVReference
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        print('[Video preview] Loading and warming the research backend before readiness...',flush=True)
        reference=OpenCVReference(gpu=True)
        reference.warmup()
        with torch.inference_mode():engine.model.arcface(torch.zeros((2,3,128,128),device=engine.device))
        app.state.research_reference=reference

    # Store in app state
    app.state.engine = engine
    app.state.gpu_name = gpu_name
    app.state.ready = True
    app.state.inference_lock = asyncio.Lock()
    if os.environ.get('ML_ENABLE_MEDIA_DETECTION') == '1':
        from ml.detection.pretrained import PretrainedDetectors
        app.state.media_detector = PretrainedDetectors()
        candidate = REPO_ROOT / 'ml/detection/swap_domain_robust'
        if (candidate / 'best.pt').exists() and (candidate / 'results.json').exists():
            app.state.media_detector.face.load_state_dict(torch.load(candidate/'best.pt',map_location='cpu',weights_only=True),strict=True)
            app.state.media_detector.face_model_name = 'Xception swap-domain robust research candidate'
        # Warm both models before declaring readiness; silence does not warm AASIST.
        from PIL import Image
        import numpy as np
        app.state.media_detector.video_score(Image.open(dummy_t).convert('RGB'))
        app.state.media_detector.audio_score(np.random.default_rng(42).normal(0,.03,64600).astype('float32'),16000)

    yield

    print("[Phase 6G API] Shutting down service...", flush=True)
    app.state.ready = False
    await close_video_peers(app)
    if hasattr(app.state,'media_detector'):app.state.media_detector.close()
    from ml.api.video_preview import close_entry
    for entry in getattr(app.state,'video_sessions',{}).values():close_entry(entry)
    if hasattr(app.state,'video_sessions'):app.state.video_sessions.clear()


app = FastAPI(
    title="Hackathena Phase 6G ML Inference API",
    description="Localhost FastAPI ML service hosting the Phase 6G neural face-swap engine",
    version="1.0.0",
    lifespan=lifespan
)
app.include_router(video_preview_router)
app.include_router(media_detection_router)
app.include_router(webrtc_video_router)

# CORS configuration for localhost Vite and Express origins (NO wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5001",
        "http://127.0.0.1:5001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns GPU status, model identifier, and readiness.
    """
    is_ready = getattr(app.state, "ready", False)
    gpu_name = getattr(app.state, "gpu_name", "Unknown GPU")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "initializing",
                "model": "phase6g",
                "device": device_name,
                "gpu": gpu_name,
                "ready": False
            }
        )

    return {
        "status": "ok",
        "model": app.state.engine.model_variant,
        "checkpoint": Path(app.state.engine.checkpoint_path).name,
        "checkpoint_sha256": app.state.engine.checkpoint_sha256,
        "correspondence": app.state.engine.correspondence,
        "video_backend": os.environ.get('ML_VIDEO_BACKEND','trained-model'),
        "video_preview_enabled": os.environ.get('ML_ENABLE_VIDEO_PREVIEW')=='1',
        "webrtc_video_enabled": os.environ.get('ML_ENABLE_WEBRTC_VIDEO')=='1',
        "media_detection_enabled": hasattr(app.state, 'media_detector'),
        "media_detection_validated_for_live_calls": False,
        "device": device_name,
        "gpu": gpu_name,
        "ready": True
    }


def validate_file(file: UploadFile, label: str):
    """Validates file mime type, extension, and presence."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{label} image filename is empty."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}' for {label}. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}"
        )

    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{file.content_type}' for {label}. Allowed: {sorted(list(ALLOWED_MIME_TYPES))}"
        )


@app.post("/infer")
async def run_inference(
    source: UploadFile = File(..., description="Source face image (provides identity)"),
    target: UploadFile = File(..., description="Target face image (provides pose, expression, and background)")
):
    """
    Phase 6G Face-Swap Inference Endpoint:
    Receives source and target images via multipart/form-data.
    Serializes inference execution (1 GPU job at a time).
    Returns swapped composite, mask, aligned source, and full diagnostic metrics.
    """
    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phase 6G model engine is not ready."
        )

    # 1. Validate upload formats
    validate_file(source, "Source")
    validate_file(target, "Target")

    # 2. Acquire GPU inference lock (Concurrency Limit: strictly 1 GPU inference at a time)
    lock: asyncio.Lock = app.state.inference_lock
    if lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Inference engine is currently busy. Single GPU worker limit enforced."
        )

    await lock.acquire()
    try:
        # Read file contents into memory and check size limit
        source_bytes = await source.read(MAX_FILE_SIZE_BYTES + 1)
        target_bytes = await target.read(MAX_FILE_SIZE_BYTES + 1)

        if len(source_bytes) == 0 or len(target_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source or Target file is empty."
            )

        if len(source_bytes) > MAX_FILE_SIZE_BYTES or len(target_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded image exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            )

        # 3. Create isolated temporary directory for inference assets
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_p = Path(temp_dir)
            src_ext = Path(source.filename).suffix.lower() or ".png"
            tgt_ext = Path(target.filename).suffix.lower() or ".png"

            src_file_path = temp_p / f"src_{uuid.uuid4().hex[:8]}{src_ext}"
            tgt_file_path = temp_p / f"tgt_{uuid.uuid4().hex[:8]}{tgt_ext}"

            with open(src_file_path, "wb") as f:
                f.write(source_bytes)
            with open(tgt_file_path, "wb") as f:
                f.write(target_bytes)

            # 4. Run inference synchronously in worker thread to prevent event-loop blocking
            engine: Phase6GInferenceEngine = app.state.engine

            def execute_sync():
                return engine.infer(
                    source_path=str(src_file_path),
                    target_path=str(tgt_file_path),
                    output_dir=str(temp_p),
                    pair_prefix="swap"
                )

            try:
                result = await run_worker(execute_sync)
            except (NoFaceDetectedError, FaceDetectionError) as fde:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Face detection failed during preprocessing: {str(fde)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference execution failed: {str(e)}"
                )

            # 5. Read output images into Base64 data URLs
            output_paths = result.get("output_paths", {})
            comp_path = output_paths.get("composite")
            mask_path = output_paths.get("mask")
            aln_path  = output_paths.get("aligned_source")
            grid_path = output_paths.get("grid")

            comp_b64 = image_data_url(comp_path)
            mask_b64 = image_data_url(mask_path)
            aln_b64 = image_data_url(aln_path)
            grid_b64 = image_data_url(grid_path)

            # 6. Format clean JSON response
            metrics = result["metrics"]
            perf = result["performance"]

            return {
                "status": "ok",
                "swapped_image": comp_b64,
                "mask_image": mask_b64,
                "aligned_source_image": aln_b64,
                "comparison_grid_image": grid_b64,
                "metadata": {
                    "model": engine.model_variant,
                    "device": perf["device"],
                    "gpu": app.state.gpu_name,
                    "latency_ms": round(perf["total_pipeline_latency_ms"], 2),
                    "network_latency_ms": round(perf["network_latency_ms"], 2),
                    "landmark_error": round(metrics["landmark_error_px"], 2) if metrics["landmark_error_px"] is not None else None,
                    "face_detected": bool(metrics["face_redetected"]),
                    "mask_mean": round(metrics["mask_mean"], 4),
                    "identity_gain": round(metrics["A_minus_C_gain"], 4),
                    "A_cosine_source_composite": round(metrics["A_cosine_source_composite"], 4),
                    "B_cosine_target_composite": round(metrics["B_cosine_target_composite"], 4),
                    "C_cosine_source_target": round(metrics["C_cosine_source_target"], 4),
                    "D_cosine_source_swap": round(metrics["D_cosine_source_swap"], 4)
                }
            }

    finally:
        lock.release()
