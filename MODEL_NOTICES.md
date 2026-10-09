# Research model notices

This is an experimental noncommercial research release. A public repository does not grant commercial model rights or certify detection accuracy. Source code and model files have separate terms; retain all upstream attribution and notices.

- InsightFace pretrained swap/identity assets (`inswapper_128.onnx`, `w600k_r50.onnx`, ArcFace iResNet-50) are restricted to noncommercial research. See https://github.com/deepinsight/insightface and contact the authors for model licensing, including commercial use.
- The Phase 6G checkpoint is project-trained on CelebA and includes pretrained identity tensors. It is supplied for noncommercial research under the applicable training-data and upstream model restrictions. It is not a claim of unrestricted ownership of all tensors.
- DeepfakeBench Xception and its project research adaptation retain CC-BY-NC-4.0 restrictions and third-party notices. See https://github.com/SCLBD/DeepfakeBench and `ml/detection/assets/DeepfakeBench.LICENSE`.
- NAVER AASIST architecture/weights retain the upstream MIT license and attribution: https://github.com/clovaai/aasist, `ml/detection/assets/AASIST.LICENSE`, and `AASIST.NOTICE`.
- Exact detector provenance, upstream commits, and checksums are recorded in `ml/detection/assets/provenance.json`. Release asset hashes and expected paths are in `ml/model-manifest.json`.
- CelebA training/test images are not redistributed. Obtain permitted research data from its official provider if you need the six preset test images. The application's existing demo portrait is used for startup when those images are absent.

Do not use these weights commercially without appropriate permissions. The downloader asks users to acknowledge this research scope; that acknowledgement is not a substitute for upstream licenses.
