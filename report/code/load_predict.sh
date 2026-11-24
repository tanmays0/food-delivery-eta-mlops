#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
N="${1:-40}"

payload='{"distance_km":4.5,"prep_time_p50_min":18,"courier_load":1,"zone_order_rate":42,"hour_of_day":19,"day_of_week":5,"is_rain":0,"traffic_index":1.3,"time_since_accept_min":3}'

echo "Sending ${N} predictions to ${BASE_URL}/predict ..."
for i in $(seq 1 "${N}"); do
  curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${payload}" >/dev/null
  sleep 0.15
done
echo "Done."
