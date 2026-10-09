# Research release validation

Validated 2026-10-09 on Windows, Python 3.12, Node.js 24, CUDA 12.8, and an NVIDIA RTX 5060 Laptop GPU.

- Lint, TypeScript checks, and frontend/backend production builds passed.
- All 11 JavaScript suites passed.
- All seven model files matched the published SHA-256 manifest.
- Python runtime/WebRTC dependencies resolved with `pip --dry-run --ignore-installed`, without relying on already-installed distributions.
- A separate checkout of the committed source built and started on separate ports without CelebA data. Existing verified model files and installed dependencies were reused for this smoke test; a completely independent GPU/package installation was not performed.
- The separate service passed all three Phase 6G golden input/output tests, malformed-upload handling, unavailable-upstream handling, and concurrent-request rejection.
- Actual received-face anomaly inference passed using the repository's existing astronaut demo portrait. Health confirmed full-resolution Phase 6G, neural video preview, WebRTC processing, and media detection enabled.
- Recognizable API-token/private-key patterns and environment-file paths were scanned in publishable source and Git history; no matching secrets were found. This scan cannot guarantee that every possible secret format is absent.
- Checksum-pinned third-party detector architecture/config/license files were preserved byte for byte. `.gitattributes` prevents checkout line-ending conversions from breaking provenance checks.

Limits: Windows-only setup has been validated; other platforms and a containerized GPU deployment have not. CelebA preset test images must be obtained separately. Detection scores are experimental, model terms restrict the research release, and the production gaps documented in `PRODUCTION_CLEANUP_REPORT.md` remain applicable.
