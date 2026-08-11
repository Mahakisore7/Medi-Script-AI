# Phase 6 — Application & Deployment (Weeks 8–13, start early in parallel)

**Owner:** Pod E (Product). **Goal:** a working web app a pharmacist could actually use — upload, view structured results, correct flagged fields, export. **Start this in week 4, against a mocked pipeline** — don't wait for Phase 5 to finish. This is the biggest schedule saver available to a 4+ person team.

## 6.1 Mock the pipeline first

```python
# services/api/app/mock_pipeline.py — delete once Phase 5's real pipeline lands
def mock_run(image_bytes: bytes) -> StructuredPrescription:
    return StructuredPrescription(
        patient_name=FieldOutput(value="Jane Doe", confidence=0.95, needs_review=False),
        medicine_and_dosage=FieldOutput(value="Paracetamol 1+1+1", confidence=0.62, needs_review=True),
        ...
    )
```
Build the entire backend and frontend against this mock, sharing the Pydantic schema from Phase 5 (agree on it in week 3–4 even before the real pipeline exists) — then swap `mock_run` for `PrescriptionPipeline.run` in week 9–11 as a one-line change.

## 6.2 Backend — FastAPI

```bash
cd services/api
uv init && uv add fastapi uvicorn sqlalchemy alembic psycopg2-binary python-multipart
```
```python
# services/api/app/main.py
from fastapi import FastAPI, UploadFile
app = FastAPI()

@app.post("/prescriptions")
async def create_prescription(file: UploadFile):
    image_bytes = await file.read()
    result = pipeline.run(image_bytes)   # or mock_run early on
    record = save_to_db(result)
    return record

@app.patch("/prescriptions/{id}")
async def correct_field(id: int, field: str, value: str):
    return update_db_field(id, field, value)

@app.get("/prescriptions/{id}/export")
async def export(id: int, format: str = "json"):
    ...
```
FastAPI auto-generates OpenAPI docs at `/docs` — genuinely useful for a 4+ person team since the frontend dev can explore and test the API without asking the backend dev every time.

## 6.3 Database — PostgreSQL + Alembic migrations

```bash
uv run alembic init migrations
```
Schema (`prescriptions` table): one row per record, one column per field's `value`/`confidence`/`needs_review`, plus `created_at`, `corrected_by`, `corrected_at` for the human-review audit trail (this audit trail is also a real privacy/accountability practice worth mentioning in your report).

Run locally via docker-compose (below) rather than installing Postgres natively on 4 different laptops.

## 6.4 Frontend — React + pnpm

```bash
cd apps/web
pnpm create vite . --template react-ts
pnpm add @tanstack/react-query axios tailwindcss
```
Minimum screens: (1) upload (camera or file), (2) results view showing the structured record with confidence-colored fields, (3) inline correction for `needs_review` fields, (4) export button. Keep it to these four — a polished MVP beats a half-built five-screen app for a semester demo.

## 6.5 Docker + docker-compose (local full-stack dev)

```yaml
# infra/docker-compose.yml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: devpassword
    ports: ["5432:5432"]
  api:
    build: ../services/api
    ports: ["8000:8000"]
    depends_on: [db]
  web:
    build: ../apps/web
    ports: ["5173:5173"]
```
`docker compose up` gives every team member the identical stack, regardless of what's installed on their own machine — the actual point of containerization, not just a buzzword for the report.

## 6.6 Deployment (keep it cheap, you have limited credits)

For a semester-project demo, you don't need Kubernetes:
- **Backend + DB:** a single AWS EC2 `t3.small` (or Elastic Beanstalk if you want less manual server config) is enough for demo-scale traffic — no GPU needed here since inference can call out to wherever the model is hosted, or run the model on a small CPU-only instance if latency isn't critical for the demo.
- **Frontend:** deploy free on Vercel or Netlify (both have generous free tiers, no need to spend AWS credits on static hosting).
- **Model serving:** if inference is too slow on CPU for a live demo, either (a) run the model on the same EC2 instance if it has a GPU, or (b) pre-compute results for your demo dataset and cache them — a completely reasonable and common thing to do for an academic demo, just be upfront about it in your report.

## 6.7 Basic privacy/security practices (address the gap flagged in your synopsis review)

- HTTPS everywhere (use a reverse proxy like Caddy or nginx with Let's Encrypt, or just Vercel/EC2's built-in options)
- Encrypt the database at rest (RDS/managed Postgres gives you this by default; self-managed Postgres needs explicit configuration)
- Don't log raw patient-identifying fields in plaintext application logs
- Basic auth on the app (even a simple shared login is better than a fully open endpoint for anything touching patient-like data, even synthetic)

## Definition of done for Phase 6

- [ ] Backend + frontend working end-to-end against the mock pipeline by week 5
- [ ] Real pipeline swapped in by week 11
- [ ] docker-compose stack runs identically on all team members' machines
- [ ] Deployed instance reachable via a public URL for your final demo
- [ ] Basic security practices (HTTPS, encryption at rest, no plaintext PII in logs) in place
