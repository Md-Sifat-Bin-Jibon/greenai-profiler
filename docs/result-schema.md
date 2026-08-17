# Result schema

`schema_version` is currently `"1.0"`.

Top-level keys:

```json
{
  "schema_version": "1.0",
  "metadata": {},
  "model": {},
  "hardware": {},
  "benchmark": {
    "latency": {},
    "memory": {},
    "energy": {},
    "config": {}
  },
  "layers": {},
  "bottlenecks": {},
  "recommendations": {},
  "accuracy": null,
  "green_score": {},
  "extra": {}
}
```

Future versions should keep readers backward compatible or provide migrations.
