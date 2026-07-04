# Contributing

This is primarily a personal research project, but issues, discussions, and PRs are welcome.

## Development setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and git — that's all;
   uv installs the correct Python itself.
2. Clone and sync:

   ```bash
   git clone <repo-url> && cd neural-ca
   uv sync
   uv run pre-commit install   # auto lint/format on every commit
   ```

3. Verify: `uv run python scripts/env_report.py && uv run pytest`

`uv sync` selects the right PyTorch build automatically (CUDA 12.8 wheels on Windows, CPU wheels
on Linux/CI). Dependencies are pinned in `uv.lock` — never edit it by hand; change
`pyproject.toml` and re-run `uv sync`.

## Quality gates (run before pushing)

```bash
uv run ruff format .   # formatter
uv run ruff check .    # linter (add --fix for autofixes)
uv run pytest          # tests
```

CI runs exactly these on every push/PR — green locally means green in CI.

## Conventions

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `perf:` — scope optional (`feat(train): …`).
- **Branches**: `main` stays green; work on `feat/<slug>`, `fix/<slug>`, `docs/<slug>`.
- **Code style** is enforced by ruff (line length 100). Type-hint public functions. In tensor
  code, make shapes visible: einops ops or `# (batch, seq, dim)` comments.
- **Tests**: new behaviour needs a test; bug fixes need a regression test.
- **Never commit** datasets or model weights — pre-commit blocks files > 1 MB; use fetch
  scripts (`scripts/`) and external storage instead.

## Documentation duties (docs-as-you-go)

Every meaningful work session leaves a trace:

1. **Devlog** — a dated entry in `docs/devlog/`: what was attempted, what broke, how it was resolved.
2. **ADR** — any hard-to-reverse decision (framework, architecture, data source) gets a file in
   `docs/decisions/`, following the existing format.
3. **CHANGELOG** — user-visible changes go under `[Unreleased]` in `CHANGELOG.md`.
