# Edge device support

Direct Android/Raspberry Pi instrumentation is not implemented in this release.

## Import protocol (planned)

External collectors should emit JSON compatible with `schema_version: "1.0"`,
including:

- device model / SoC
- collection tool version
- latency / energy fields with explicit `status`
- notes describing the on-device methodology

Desktop runs must not fabricate edge measurements.
