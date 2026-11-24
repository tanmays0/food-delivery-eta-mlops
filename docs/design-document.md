# ML System Design: Food Delivery ETA Predictor

**Course Project:** P2 — ML System Design for a Real-World Product  
**Student:** Tanmay Shinde | **ID:** ADT23SOCB1617  
**Architecture diagram:** [`../diagrams/architecture-diagram.png`](../diagrams/architecture-diagram.png)

---

## 1. Project Overview and Problem Statement

Food-delivery platforms must show a trustworthy Estimated Time of Arrival (ETA) at checkout and update it as preparation, traffic, and courier location change. End-to-end ETA combines restaurant preparation, wait for a delivery partner, road travel, and drop-off. Simple averages break under peak demand, weather, and heterogeneous restaurants.

This document designs an ML system that predicts delivery duration (minutes) from tabular operational features. Independent microservices on AWS provide real-time inference for customer-facing ETAs and batch pipelines for feature computation and retraining. Numeric goals below are **design targets**, not measured lab results.

---

## 2. System Requirements

**Functional:** Return ETA (minutes, optional confidence) from order and context features; refresh on GPS/status/traffic events; log predictions vs actuals; fall back to a heuristic (distance-based travel + median prep) if the model or feature store fails.

| Metric | Design target | Notes |
|--------|---------------|--------|
| API latency (p95) | ≤ 100 ms | Feature fetch + XGBoost inference |
| Throughput | 500–1000 RPS peak | City-scale assumption |
| Availability | 99.9% (ETA API) | Multi-AZ EKS and RDS |
| Accuracy | MAE ≤ 4–5 min; track RMSE | Time-based holdout |
| Scalability | HPA on CPU / RPS | ETA and Location services |
| Fault tolerance | Multi-AZ; Kafka retention; heuristic fallback | Per-service isolation |

**Evaluation.** Use MAE and RMSE on predicted vs actual duration with a **time-based holdout** (e.g., last two weeks) to avoid leakage. After delivery completes, log residual = predicted − actual for online monitoring. MAPE is secondary (short trips inflate percentages).

---

## 3. Data Sources and Data Warehouse Design

| Source | Examples | Role |
|--------|----------|------|
| Historical deliveries | Accept → pickup → drop times | Labels |
| Restaurant prep | Menu complexity, prep distributions | Prep features |
| Partner GPS | Lat/lon, speed, status | Travel / dispatch |
| Distance / routes | Road distance, maps ETA | Travel signal |
| Traffic / weather | Congestion, rain | Travel and prep |
| Demand / calendar | Zone order rate, hour, dow, holidays | Queues and cycles |

**Pipeline:** Sources → Kafka / batch extracts → transforms → **S3 Data Lake** (Bronze raw → Silver cleaned → Gold aggregates) → **Amazon Redshift** (SQL analytics / training extracts) → **Feature Store** (offline tables + Redis online).

- **Streaming:** Orders, GPS, and traffic on Kafka (`orders`, `gps`, `traffic`, `eta_updates`) refresh online features quickly.
- **Batch:** Nightly CDC/snapshots from PostgreSQL and weather dumps land in S3 Bronze, then Silver/Gold and Redshift.

S3 stores cheap history; Redshift joins deliveries with weather, demand, and routes for training sets. Redis serves online feature vectors within the latency budget; matching offline definitions limits train–serve skew.

---

## 4. ML Training Pipeline

1. **Preprocess:** Drop incomplete deliveries; clip extremes; impute rare missing weather; encode categoricals.  
2. **Features:** Distance; restaurant prep p50/p90; courier load; zone demand; hour/dow; rain; traffic index; time since acceptance.  
3. **Baselines:** Linear Regression and Random Forest for a sanity floor.  
4. **Production model:** **XGBoost** regressor — strong tabular accuracy, fast CPU inference, feature importance for debugging.  
5. **Validate / tune:** Time-based split; search depth, learning rate, subsample; select by MAE (report RMSE).  
6. **Registry:** **MLflow** logs params, metrics, artifacts; promote only after offline gates.  
7. **Deploy:** ETA pods pull the production artifact (EKS rolling update).  
8. **Retrain:** Weekly schedule plus triggers when online MAE or drift breaches thresholds.

Deep models are unnecessary here: features are tabular, and GPUs would add cost and ops without a clear ETA gain.

---

## 5. Model Serving Architecture

| Mode | Workload | Why |
|------|----------|-----|
| **Real-time** | New-order ETA; mid-trip updates | Sub-second UX |
| **Batch** | Daily features; aggregates; retrain | Heavy history joins |

**Hybrid (selected).** FastAPI serves synchronous REST inference. Batch jobs refresh features and retrain. Kafka triggers async ETA recalculation on GPS/status changes without blocking checkout.

**Online path:** Client → API Gateway → Order Management → Feature Engineering (Redis + request context) → ETA Prediction (XGBoost) → `{ eta_minutes, model_version, fallback }` → apps. Predictions and later actuals feed monitoring.

---

## 6. Microservices and Cloud Infrastructure

Docker containers on **Amazon EKS** enable independent scale and rolling deploys.

| Component | Technology | Purpose |
|-----------|------------|---------|
| Edge | AWS API Gateway | Auth, rate limits, routing |
| Order Management | Service + **PostgreSQL (RDS)** | Transactional orders |
| Location Tracking | Service | GPS ingest → Kafka |
| Feature Engineering | Service | Stream/batch → Redis / lake |
| ETA Prediction | **FastAPI** + XGBoost | Real-time inference |
| Event bus | **Kafka** (MSK) | Decouple producers/consumers |
| Lake / warehouse | **S3** + **Redshift** | History and analytics |
| Feature store | **Redis** (+ offline tables) | Low-latency features |
| Registry | **MLflow** | Versions and promotion |
| Observability | **Prometheus, Grafana, CloudWatch** | Metrics and alarms |

**REST** for place-order / ETA reads; **Kafka** for high-volume GPS and status fan-out.

---

## 7. Monitoring and Maintenance

Monitor API latency and RPS (alert if p95 > 100 ms), online MAE/RMSE after deliveries complete, data quality (nulls, GPS freshness), data drift (feature distribution shift), and model drift (rising residual error by zone). Kubernetes probes and CloudWatch cover service/infra health.

**Response:** On MAE or drift breach, review feature pipelines first, then retrain via MLflow. If the candidate fails gates, keep the prior production model and lean on heuristic fallback until a healthy version is promoted. Weekly scheduled retrain remains the default cadence.

---

## 8. Architectural Trade-offs and Justifications

| Decision | Selected | Alternative | Justification |
|----------|----------|-------------|---------------|
| Processing | Hybrid real-time + batch | Batch- or stream-only | Immediate ETA at order time; batch for training/features |
| Architecture | Microservices on EKS | Monolith | Independent scale for GPS vs ETA; failure isolation |
| Integration | REST + Kafka | REST-only or events-only | Sync checkout UX; async GPS flood |
| Model | XGBoost (+ LR/RF) | RF-only or deep nets | Accuracy / latency / ops balance on tabular data |
| Storage | PostgreSQL + S3/Redshift | SQL-only or lake-only | ACID orders + cheap history/analytics |
| Inference | CPU | GPU | Tree models; lower cost |
| Orchestration | Kubernetes (EKS) | Single VM / Compose-only | Autoscale multi-service production story |

---

## 9. Conclusion

The design pairs real-time FastAPI/XGBoost inference with Kafka feature updates, PostgreSQL for orders, and S3/Redshift feeding MLflow retraining. Targets of ≤100 ms p95, hundreds of RPS, and MAE around 4–5 minutes guide capacity and monitoring. Hybrid serving, EKS microservices, and drift-triggered retrain loops keep ETAs useful as demand and roads change—meeting the assignment’s need for a complete, trade-off-aware ML system design.
