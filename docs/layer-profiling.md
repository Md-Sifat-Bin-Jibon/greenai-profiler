# Layer profiling

Leaf modules are instrumented with forward pre/post hooks. Parent containers are
skipped to reduce double-counting.

Reported fields:

- layer name and type
- average latency across iterations
- percent of summed leaf latency
- parameter count (non-recursive)
- input/output shapes when tensors are available
- dtype when parameters exist

Energy per layer is marked unavailable unless a future backend can isolate it.
