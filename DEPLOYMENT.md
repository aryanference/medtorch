# MedTorch Deployment Guide

This project is configured for package publishing and documentation deployment through GitHub Actions.

## Release strategy

- **TestPyPI publish**: manual trigger via `workflow_dispatch` on the `Publish` workflow.
- **PyPI publish**: automatic on git tags matching `v*` (for example `v1.2.1`).
- **Docs deploy**: automatic to GitHub Pages from `main` via the `Documentation` workflow.

## Pre-release checklist

- Bump version in `pyproject.toml` and `src/torchio/__init__.py`.
- Confirm package metadata is correct (`name`, `description`, URLs, Python version).
- Run tests locally (or in CI) and ensure all required checks pass.
- Build artifacts locally:

```bash
uv build
```

- Optional verification before publishing:

```bash
uv run -- python -m pip install dist/*.whl
```

## TestPyPI dry run

1. Open GitHub Actions.
2. Run `Publish` workflow manually.
3. Confirm package appears on TestPyPI and can be installed.

## Production release (PyPI)

1. Create and push a version tag:

```bash
git tag v1.2.1
git push origin v1.2.1
```

2. Wait for the `Publish` workflow to finish.
3. Verify package on PyPI:

```bash
pip install medtorch
```

## Post-release checks

- Confirm CLI entry points resolve:
  - `medtorch-info --help`
  - `medtorch-transform --help`
- Confirm docs build and publish status on GitHub Pages.
- Validate install/import smoke test:

```bash
python -c "import torchio as tio; print(tio.__version__)"
```
