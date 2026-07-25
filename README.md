# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, building an end-to-end pipeline for handwritten medical prescription recognition.

## Branch: `main` (Phase 0 Foundation)

This branch contains the foundational **Phase 0** repository scaffolding. It establishes the monorepo structure, Python tooling (`uv`), and strict GitHub CI/CD standards according to the `docs_build_guide`.

### Directory Structure
- `ml/`: Python machine learning pipeline, training, and evaluation.
- `api/`: FastAPI backend (Phase 6).
- `web/`: React + Vite frontend (Phase 6).
- `docs/`: Build guide documentation and architectural diagrams.
- `data/`: Local testing datasets (gitignored).

### Local Setup

We use `uv` for reproducible, fast Python environment management.

1. **Install uv**:
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```
2. **Install dependencies**:
   ```bash
   cd ml
   uv sync
   ```
3. **Run Linting**:
   ```bash
   uv run --project ml ruff check .
   uv run --project ml pytest ml/tests -v
   ```

*(Note: The old IAM pretraining proof-of-concept can be found in the `iam_pretraining_poc/` directory).*
