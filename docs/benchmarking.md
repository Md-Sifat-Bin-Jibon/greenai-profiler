# Benchmarking methodology

## Latency

1. Construct a synthetic input with the requested batch size and shape.
2. Run `warmup` iterations (excluded from statistics).
3. For each timed iteration:
   - synchronize CUDA (if device is CUDA) before starting the timer
   - run forward pass
   - synchronize CUDA before stopping the timer
4. Report mean, median/P50, P95, P99, min, max, stddev.
5. Throughput = `batch_size / mean_latency_seconds`.

## Common pitfalls avoided

- Timing GPU work without synchronization
- Including model load / first-compile in measured iterations
- Reporting a single run as a stable latency

## Reproducibility fields

Results JSON records hardware, Python/PyTorch versions, precision/dtypes,
batch size, input shape, warmup/iteration counts, timestamp, and method notes.
