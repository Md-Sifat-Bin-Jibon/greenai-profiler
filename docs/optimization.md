# Optimization

Helpers in this release:

- FP16 conversion (`greenai.optimization.fp16.to_fp16`)
- Dynamic INT8 for Linear layers (CPU-oriented)
- Unstructured magnitude pruning for Linear layers
- Evidence-based textual recommendations from profile JSON

The CLI does not silently rewrite models. Compare results with `greenai compare`.
