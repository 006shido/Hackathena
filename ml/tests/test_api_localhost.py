"""
Test Suite for Phase 6G Localhost FastAPI Service
Tests:
  1. GET  /health (GPU, readiness, metadata)
  2. POST /infer with 3 genuine CelebA cross-ID validation pairs
  3. POST /infer error handling (invalid file format)
"""

import os
import sys
import time
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests

API_URL = os.environ.get("TEST_ML_URL", "http://127.0.0.1:8001")
OUTPUT_DIR = Path("ml/inference/outputs/api_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    {
        "name": "Pair 1 (004831 -> 004865)",
        "source": "ml/data/celeba/img_align_celeba/004831.jpg",
        "target": "ml/data/celeba/img_align_celeba/004865.jpg"
    },
    {
        "name": "Pair 2 (004931 -> 004842)",
        "source": "ml/data/celeba/img_align_celeba/004931.jpg",
        "target": "ml/data/celeba/img_align_celeba/004842.jpg"
    },
    {
        "name": "Pair 3 (004893 -> 004925)",
        "source": "ml/data/celeba/img_align_celeba/004893.jpg",
        "target": "ml/data/celeba/img_align_celeba/004925.jpg"
    }
]


def test_health():
    print("\n--- 1. Testing GET /health ---")
    resp = requests.get(f"{API_URL}/health", timeout=10)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print("Response JSON:", data)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert data["status"] == "ok"
    assert data["model"] == os.environ.get("TEST_ML_MODEL", "fullres")
    assert data["device"] == "cuda"
    assert data["ready"] is True
    print("GET /health PASSED!")
    return data


def test_infer_pair(pair_idx: int, pair: dict):
    print(f"\n--- Testing POST /infer: {pair['name']} ---")
    src_path = Path(pair["source"])
    tgt_path = Path(pair["target"])

    assert src_path.exists(), f"Missing source: {src_path}"
    assert tgt_path.exists(), f"Missing target: {tgt_path}"

    with open(src_path, "rb") as f_src, open(tgt_path, "rb") as f_tgt:
        files = {
            "source": (src_path.name, f_src, "image/jpeg"),
            "target": (tgt_path.name, f_tgt, "image/jpeg")
        }
        t0 = time.time()
        resp = requests.post(f"{API_URL}/infer", files=files, timeout=60)
        round_trip_ms = (time.time() - t0) * 1000.0

    print(f"HTTP Status: {resp.status_code} (Round-trip: {round_trip_ms:.1f} ms)")
    assert resp.status_code == 200, f"Inference failed with status {resp.status_code}: {resp.text}"

    data = resp.json()
    assert data["status"] == "ok"
    assert "swapped_image" in data and len(data["swapped_image"]) > 100
    assert "metadata" in data

    meta = data["metadata"]
    print(f"  Model:           {meta['model']}")
    print(f"  Device:          {meta['device']} ({meta['gpu']})")
    print(f"  Pipeline Latency:{meta['latency_ms']} ms")
    print(f"  Network Latency: {meta['network_latency_ms']} ms")
    print(f"  Face Detected:   {meta['face_detected']}")
    print(f"  Landmark Error:  {meta['landmark_error']} px")
    print(f"  Mask Mean:       {meta['mask_mean']}")
    print(f"  Identity Gain:   {meta['identity_gain']:+.4f}")
    print(f"  A (Src-Comp):    {meta['A_cosine_source_composite']:.4f}")
    print(f"  C (Src-Tgt):     {meta['C_cosine_source_target']:.4f}")

    # Decode and save image to verify validity
    prefix = "data:image/png;base64,"
    assert data["swapped_image"].startswith(prefix)
    b64_str = data["swapped_image"][len(prefix):]
    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(BytesIO(img_bytes))
    assert pil_img.size == (128, 128), f"Expected 128x128, got {pil_img.size}"

    save_path = OUTPUT_DIR / f"pair_{pair_idx + 1}_composite.png"
    pil_img.save(save_path)
    print(f"  Saved composite image to: {save_path}")

    # Also save comparison grid if returned
    if data.get("comparison_grid_image") and data["comparison_grid_image"].startswith(prefix):
        grid_b64 = data["comparison_grid_image"][len(prefix):]
        grid_bytes = base64.b64decode(grid_b64)
        grid_pil = Image.open(BytesIO(grid_bytes))
        grid_save = OUTPUT_DIR / f"pair_{pair_idx + 1}_grid.png"
        grid_pil.save(grid_save)
        print(f"  Saved grid image to: {grid_save}")

    assert meta["face_detected"] is True, "Output face was not redetected!"
    assert meta["latency_ms"] < 1000.0, f"Latency unexpectedly high: {meta['latency_ms']} ms"
    print(f"Pair {pair_idx + 1} PASSED!")

    return {
        "pair": pair["name"],
        "round_trip_ms": round_trip_ms,
        "pipeline_latency_ms": meta["latency_ms"],
        "network_latency_ms": meta["network_latency_ms"],
        "landmark_error_px": meta["landmark_error"],
        "identity_gain": meta["identity_gain"],
        "face_detected": meta["face_detected"]
    }


def test_invalid_upload():
    print("\n--- 3. Testing POST /infer validation on invalid input ---")
    files = {
        "source": ("test.txt", BytesIO(b"not an image"), "text/plain"),
        "target": ("test.txt", BytesIO(b"not an image"), "text/plain")
    }
    resp = requests.post(f"{API_URL}/infer", files=files, timeout=10)
    print(f"HTTP Status on invalid file: {resp.status_code}")
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    print("Invalid upload rejection PASSED!")


def main():
    print("==================================================")
    print("PHASE 6G FASTAPI LOCALHOST VERIFICATION SUITE")
    print("==================================================")

    # 1. Health check
    test_health()

    # 2. Test 3 genuine CelebA validation pairs
    results = []
    for idx, pair in enumerate(PAIRS):
        res = test_infer_pair(idx, pair)
        results.append(res)

    # 3. Test validation error handling
    test_invalid_upload()

    print("\n==================================================")
    print("ALL API TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
    print(f"{'Pair':<30} | {'Roundtrip':<10} | {'Pipeline':<10} | {'Net GPU':<10} | {'LM Error':<10} | {'Gain A-C':<10} | {'Face OK'}")
    print("-" * 95)
    for r in results:
        print(f"{r['pair']:<30} | {r['round_trip_ms']:>8.1f}ms | {r['pipeline_latency_ms']:>8.1f}ms | {r['network_latency_ms']:>8.1f}ms | {r['landmark_error_px']:>8.2f}px | {r['identity_gain']:>+9.4f} | {r['face_detected']}")
    print("-" * 95)


if __name__ == "__main__":
    main()
