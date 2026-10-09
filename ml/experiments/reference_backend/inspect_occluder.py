"""Measure encoded changes inside a manually annotated occluder.

This is an artifact diagnostic, not an automatic segmentation or quality gate.
Comparison video columns must be original and generated, with equal widths.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--polygon', required=True, help='JSON array of original-panel pixel coordinates')
    parser.add_argument('--raw', action='store_true', help='Use saved lossless inference panels instead of encoded video')
    args = parser.parse_args()
    polygon = np.asarray(json.loads(args.polygon), dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError('Expected at least three polygon points.')
    if args.raw:
        panels = [cv2.imread(str(args.folder / f'raw_{label}_{args.frame}.png')) for label in ('original','output')]
        ok = all(panel is not None for panel in panels)
        frame = np.concatenate(panels, axis=1) if ok else None
    else:
        capture = cv2.VideoCapture(str(args.folder / 'comparison.mp4'))
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = capture.read()
        capture.release()
    if not ok or frame.shape[1] % 2:
        raise ValueError('Missing frame or unequal comparison panels.')
    width = frame.shape[1] // 2
    original, generated = frame[:, :width], frame[:, width:]
    if np.any(polygon < 0) or np.any(polygon[:, 0] >= width) or np.any(polygon[:, 1] >= frame.shape[0]):
        raise ValueError('Polygon is outside the original panel.')
    mask = np.zeros(original.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    if not mask.any():
        raise ValueError('Polygon has no interior after erosion.')
    difference = np.abs(original.astype(np.float32) - generated.astype(np.float32))
    changed = difference.max(axis=2) > 8
    report = {
        'frame': args.frame, 'polygon': polygon.tolist(), 'erosion_kernel': 9,
        'interior_pixels': int(mask.sum()),
        'mean_absolute_rgb_difference': float(difference[mask].mean()),
        'fraction_over_8_rgb_levels': float(changed[mask].mean()),
        'scope': ('Manual occluder annotation on lossless inference panels; exact pixel differences, not automatic segmentation.' if args.raw else 'Manual occluder annotation on encoded panels. Codec error is included; not an exact raw-pixel preservation test or automatic segmentation.')
    }
    heat = cv2.applyColorMap(np.clip(difference.max(axis=2) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.polylines(heat, [polygon], True, (255, 255, 255), 1)
    stem = args.folder / f'occluder_frame{args.frame}'
    cv2.imwrite(str(stem.with_suffix('.png')), np.concatenate([original, generated, heat], axis=1))
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
