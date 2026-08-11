# Phase 0 — Foundations & Setup (Weeks 1–2)

**Goal:** by the end of this phase, every team member can clone the repo, run one command to get a working environment, and the team has agreed on data strategy, architecture, and how you'll work together in git. Nothing ML-specific happens yet — this phase is infrastructure, and it's the phase most student teams skip, which is exactly why their projects fall apart in week 10.

## 0.1 Repo structure

Create a single monorepo (one repo, multiple apps) rather than separate repos per pod — with a team of 4+ working across ML/backend/frontend, a monorepo keeps integration friction low.

```
prescription-recognition/
├── ml/                        # all training/inference code
│   ├── data/                  # dataset loading, augmentation
│   ├── segmentation/          # YOLO field detector
│   ├── recognition/           # TrOCR fine-tuning
│   ├── postprocess/           # lexicon validation, multilingual mapping
│   ├── pipeline/              # unified inference pipeline (Phase 5)
│   ├── evaluation/            # metrics, benchmarking (Phase 7)
│   ├── notebooks/             # EDA, experiments (not production code)
│   ├── pyproject.toml
│   └── uv.lock
├── services/
│   └── api/                   # FastAPI backend
│       ├── app/
│       ├── pyproject.toml
│       └── uv.lock
├── apps/
│   └── web/                   # React frontend
│       ├── src/
│       ├── package.json
│       └── pnpm-lock.yaml
├── infra/                     # Dockerfiles, docker-compose, optional Terraform
├── docs/                      # these phase files + diagrams live here
├── data/                      # DVC pointers only — never real data files
│   └── .gitignore
├── .github/
│   ├── workflows/             # CI
│   └── pull_request_template.md
├── .gitignore
└── README.md
```

## 0.2 Install and initialize tooling

**uv** (Python). On Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```
Then per Python subproject (`ml/`, `services/api/`):
```bash
cd ml
uv init --python 3.11
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match your CUDA version
uv add transformers datasets accelerate evaluate jiwer rapidfuzz opencv-python ultralytics wandb dvc[s3]
uv run python -c "import torch; print(torch.cuda.is_available())"
```
`uv run <command>` runs inside the project's locked environment without manually activating a venv — use this everywhere instead of remembering to `activate`.

**pnpm** (Node), for the frontend:
```powershell
corepack enable
corepack prepare pnpm@latest --activate
cd apps/web
pnpm create vite . --template react-ts
pnpm add -D tailwindcss postcss autoprefixer
pnpm install
```

## 0.3 Git workflow — set this up before anyone writes ML code

1. Create the repo on GitHub, add all teammates as collaborators.
2. Protect `main`: Settings → Branches → require PR + 1 approval before merge, require status checks (CI) to pass.
3. Add `.github/pull_request_template.md`:
   ```markdown
   ## What changed
   ## How it was tested
   ## Screenshot / metric (if applicable)
   ## Linked issue
   ```
4. Agree on commit style now (Conventional Commits): `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`. Put it in `CONTRIBUTING.md`.
5. Set up pre-commit hooks so bad commits never land:
   ```bash
   uv add --dev pre-commit ruff
   uv run pre-commit install
   ```
   `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.6.9
       hooks:
         - id: ruff
         - id: ruff-format
   ```
6. First CI workflow, `.github/workflows/ci.yml` — lint + test on every PR:
   ```yaml
   name: CI
   on: [pull_request]
   jobs:
     lint-and-test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: astral-sh/setup-uv@v3
         - run: cd ml && uv sync && uv run ruff check . && uv run pytest
   ```

## 0.4 Cloud accounts

You have starter credits on AWS, Azure, and GCP. Don't split the project across all three — pick **one primary** (this guide assumes **AWS**) and treat the others as reserve capacity if you run low. Steps:

1. Create an IAM user (not root) with programmatic access; never use root credentials day-to-day.
2. **Set a billing alarm immediately** — CloudWatch billing alarm at, say, 50% and 90% of your credit. This is the single most important 10-minute task in this entire phase; GPU instances left running overnight is the #1 way student teams burn through cloud credits.
3. Create an S3 bucket for datasets/model artifacts: `aws s3 mb s3://<team-name>-prescription-data`.
4. Install AWS CLI, run `aws configure` on each team member's machine (or better: use a shared `.env`-style profile documented in `CONTRIBUTING.md`, never commit keys).
5. Note the service-equivalents in case you need Azure/GCP overflow later:

| Need | AWS | GCP | Azure |
|---|---|---|---|
| Object storage | S3 | Cloud Storage | Blob Storage |
| GPU VM | EC2 (g4dn/g5) | Compute Engine (T4/L4) | NC-series VM |
| Managed training | SageMaker | Vertex AI | Azure ML |

## 0.5 Experiment tracking & data versioning

```bash
uv run wandb login          # paste API key from wandb.ai (free academic tier)
uv run dvc init
uv run dvc remote add -d storage s3://<team-name>-prescription-data
```
From now on: code goes in git, data/model weights go in DVC (which stores a small pointer file in git and the real bytes in S3). This is what keeps your local 20GB free — nobody needs the full dataset on their laptop, `dvc pull` fetches only what you're actively working on.

## 0.6 Dataset strategy sign-off (do this now, in writing)

Given the semester timeline, decide and document (in `docs/data-strategy.md`) your fallback data plan rather than assuming real hospital data will materialize:

- **Primary source:** synthetic-but-realistic — recruit volunteers (classmates, family, willing clinicians) to hand-write prescriptions using a fixed template with **fabricated** patient details (never real patient PHI). This sidesteps IRB/ethics review entirely since no real patient data is involved, and you control field layout consistency, which makes annotation much faster.
- **Supplementary:** public datasets — IAM-OnDB (handwriting, not prescription-specific but useful for pretraining/augmentation) and prescription-specific sets on Roboflow Universe (search "handwritten prescription").
- **Stretch, only if a real clinic partnership becomes available with proper consent/anonymization:** real prescriptions, PHI stripped or synthetic patient fields overlaid on a real doctor's handwriting.

Write this down and get your guide/supervisor to sign off on it in week 1 — it's the assumption everything else in the timeline depends on.

## 0.7 Diagrams

Use the architecture diagram and user-flow sequence diagram from `README.md` as your starting point — they're Mermaid, so they render directly on GitHub and can live in `docs/architecture.md` versioned alongside the code (diagrams-as-code is the production practice here, not a one-off PNG in a slide deck that goes stale).

## Definition of done for Phase 0

- [ ] Repo created, structure above scaffolded, everyone has cloned it and run the environment setup successfully
- [ ] Branch protection + PR template + CI (lint+test) live and passing on a trivial PR
- [ ] AWS account set up, billing alarm active, S3 bucket created
- [ ] W&B and DVC initialized
- [ ] `docs/data-strategy.md` written and signed off by your guide
- [ ] Architecture + user-flow diagrams committed to `docs/architecture.md`
- [ ] Pod assignments agreed and written down (who owns what branch prefix)
