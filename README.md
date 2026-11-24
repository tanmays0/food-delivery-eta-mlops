# Food Delivery ETA Predictor — MLOps Project (P2)

**Student:** Tanmay Shinde · **Class:** LY-IT-SMAD · **Enrollment:** ADT23SOCB1617  
**Subject:** MLOps · **Assessment:** End Semester Jury

ML system design **and** working demo for a food-delivery Estimated Time of Arrival (ETA) predictor: training (LR / RF / XGBoost), FastAPI serving, Docker Compose, Prometheus + Grafana.

**GitHub:** https://github.com/tanmays0/food-delivery-eta-mlops

---

## Submission package

| Document | Link |
|----------|------|
| **Full project report** (theory + screenshots + code + links) | [`report/PROJECT_REPORT.md`](report/PROJECT_REPORT.md) |
| Compact design document (2–3 pages) | [`docs/design-document.md`](docs/design-document.md) |
| Architecture diagram | [`diagrams/architecture-diagram.png`](diagrams/architecture-diagram.png) |
| Screenshots | [`outputs/screenshots/`](outputs/screenshots/) / [`report/assets/`](report/assets/) |

---

## Quick start (Docker)

```bash
git clone https://github.com/tanmays0/food-delivery-eta-mlops.git
cd food-delivery-eta-mlops

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py && python src/train.py

docker compose up -d --build
./scripts/load_predict.sh 40
```

| Service | URL |
|---------|-----|
| Demo UI | http://localhost:8000/ |
| API docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (`admin` / `admin`) |
| Dashboard | http://localhost:3000/d/eta-serving/food-delivery-eta-serving |

---

## Measured results (synthetic demo)

| Model | MAE (min) | RMSE (min) | R² |
|-------|-----------|------------|----|
| Linear Regression | 3.071 | 3.927 | 0.960 |
| Random Forest | 2.427 | 3.127 | 0.975 |
| **XGBoost (production)** | **2.096** | **2.666** | **0.982** |

---

## Repository layout

```text
src/           # data generation, training, FastAPI + Prometheus metrics
static/        # demo UI
models/        # trained artifacts
monitoring/    # Prometheus + Grafana provisioning
report/        # submission-ready project report
docs/          # short design document
diagrams/      # architecture diagram
docker-compose.yml
```
