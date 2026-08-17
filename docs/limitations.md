# Limitations

- Energy measurement depends on host telemetry; many machines will report unavailable.
- CPU RSS includes the whole process, not just weights.
- Layer latency from hooks is useful for ranking, not a perfect substitute for
  vendor profilers on fused kernels.
- Green Score is a documented heuristic for relative comparison only.
- Pickle loading is dangerous; safe defaults may reject some legitimate checkpoints.
- Edge/mobile numbers must be imported from on-device collection—desktop fakes are refused.
