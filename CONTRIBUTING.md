# Contributing

```bash
git clone https://github.com/Samrdreyfuss/graphdml && cd graphdml
uv venv && uv pip install -e ".[dev]"    # or: python -m venv .venv && pip install -e ".[dev]"
pytest                                    # fast suite
pytest -m slow                            # coverage gate (about a minute)
ruff check .
```

Neo4j integration tests run against a live server:

```bash
docker compose up -d
GRAPHDML_NEO4J_URI=bolt://localhost:7687 GRAPHDML_NEO4J_PASSWORD=graphdml-demo pytest tests/test_neo4j.py
```

Work happens on the `dev` branch and reaches `main` through a pull request; `main` requires
the CI checks to pass. Please read the [code of conduct](CODE_OF_CONDUCT.md) first.

Changes that affect estimates should keep [docs/methodology.md](docs/methodology.md) accurate
and pass the [validation protocol](docs/validation.md). The benchmarks in `benchmarks/` and
`benchmarks/report.py` regenerate [docs/validation-results.md](docs/validation-results.md).
