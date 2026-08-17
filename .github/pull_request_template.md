## Summary

<!-- Why this change exists. Link related issues when applicable. -->

## Test plan

- [ ] `ruff check src tests examples`
- [ ] `ruff format --check src tests examples`
- [ ] `mypy src`
- [ ] `pytest`

## Checklist

- [ ] Energy / power numbers are measured or explicitly unavailable (never fabricated)
- [ ] Tests added or updated for behavior changes
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] Security-sensitive paths (model loading, pickle) stay opt-in and documented
