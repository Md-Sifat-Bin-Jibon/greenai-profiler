# Energy measurement

## Principle

Never fabricate energy. Never report `latency × arbitrary_power` as measured energy.

## Backends

| Backend | Method | Status label |
|---|---|---|
| `NvidiaEnergyMonitor` | Mean of NVML power samples × duration | `measured` |
| `IntelRaplMonitor` | Delta of RAPL `energy_uj` | `measured` |
| `UnsupportedEnergyMonitor` | None | `unavailable` |

### NVIDIA notes

- Requires working NVML (`nvidia-ml-py` + driver).
- Energy is approximated from power telemetry over the benchmark window.
- Idle/baseline power is **not** subtracted in this release (documented in result notes).

### RAPL notes

- Linux powercap sysfs only.
- Counter wrap is treated as unavailable rather than guessed.

## Layer energy

Exact per-layer joules are generally unavailable on commodity APIs. Layer profiles
mark energy as `unavailable` unless a future backend can justify otherwise.
