# Training Results — Food Delivery ETA Predictor

- Train samples: **6400** | Test samples: **1600**
- Selected production model: **xgboost**

| Model | MAE (min) | RMSE (min) | R² |
|-------|-----------|------------|----|
| linear_regression | 3.071 | 3.927 | 0.960 |
| random_forest | 2.427 | 3.127 | 0.975 |
| xgboost ← production | 2.096 | 2.666 | 0.982 |

## Sample predictions (XGBoost)

See `sample_predictions.csv` for full rows.

```
 distance_km  prep_time_p50_min  courier_load  zone_order_rate  hour_of_day  day_of_week  is_rain  traffic_index  time_since_accept_min  actual_delivery_min  pred_xgboost_min  abs_error_min
        1.78              23.18           0.0            54.76         14.0          2.0      0.0           0.79                   4.20                37.30         41.209999           3.91
        1.66              21.75           2.0            76.07          8.0          0.0      0.0           1.75                   5.06                44.25         44.919998           0.68
        9.57              15.13           1.0            49.69         12.0          5.0      0.0           0.66                   2.53                73.03         70.910004           2.12
       10.43              30.31           1.0            60.95         19.0          1.0      0.0           1.01                  11.55                95.00         92.870003           2.13
        3.42              31.76           3.0            78.29         21.0          5.0      0.0           1.42                   2.31                66.47         66.110001           0.35
        9.77              21.47           1.0            55.68         12.0          3.0      0.0           0.97                   9.50                85.60         85.730003           0.13
        1.60              22.34           2.0            35.35         12.0          4.0      0.0           1.17                   8.19                36.87         41.610001           4.74
        2.78              17.42           1.0            40.71         18.0          0.0      0.0           0.71                   7.45                38.11         38.299999           0.18
```

Artifacts: `/Users/shind/Desktop/TanmayShinde_ADT23SOCB1617_2/models` | Metrics: `/Users/shind/Desktop/TanmayShinde_ADT23SOCB1617_2/outputs/training_metrics.json`