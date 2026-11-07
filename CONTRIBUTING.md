# Contributing

Bug reports and focused pull requests are welcome. Please include a minimal
reproduction, use generated or redistributable test inputs, and run:

```bash
python -m unittest discover -s tests -v
ruff check flexmm tests
```

Do not commit model weights, datasets, participant data, generated outputs, or
credentials. Contributions must preserve the provenance and licensing notices
of code derived from third parties.
