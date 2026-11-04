# Phase 3 - Permutation Feature Importance

- Dataset SHA-256: `28b6991062855d3d15642e38b1aa9d1b32fb7abbb5abf5b096f1868cd706154b`
- Total rows: 2087
- Holdout samples: 418
- Baseline Macro-F1: 0.8379
- Method: Permutation Importance
- Scoring: Macro-F1
- Repeats: 30
- Random state: 42

## Feature ranking

| Rank | Feature | Mean importance | Std |
|---:|---|---:|---:|
| 1 | Giới tính (`Gender`) | 0.262609 | 0.015898 |
| 2 | Tuổi (`Age`) | 0.174352 | 0.014460 |
| 3 | Tần suất ăn rau (`FCVC`) | 0.168280 | 0.015539 |
| 4 | Tiền sử gia đình thừa cân (`family_history_with_overweight`) | 0.160557 | 0.014374 |
| 5 | Sử dụng đồ uống có cồn (`CALC`) | 0.110797 | 0.013518 |
| 6 | Số bữa ăn chính (`NCP`) | 0.076410 | 0.008677 |
| 7 | Ăn vặt giữa các bữa (`CAEC`) | 0.069572 | 0.009552 |
| 8 | Thời gian sử dụng thiết bị điện tử (`TUE`) | 0.050882 | 0.009722 |
| 9 | Lượng nước uống (`CH2O`) | 0.044865 | 0.009090 |
| 10 | Hoạt động thể chất (`FAF`) | 0.040350 | 0.010821 |
| 11 | Phương tiện di chuyển (`MTRANS`) | 0.031945 | 0.008556 |
| 12 | Thường xuyên ăn thực phẩm giàu calo (`FAVC`) | 0.027190 | 0.005162 |
| 13 | Theo dõi lượng calo (`SCC`) | 0.006609 | 0.001846 |
| 14 | Hút thuốc (`SMOKE`) | 0.000269 | 0.000807 |

## Interpretation

Permutation importance measures the decrease in Macro-F1 after randomly permuting one input feature.

A larger positive value indicates greater reliance by this fitted model on that feature.

## Limitations

- Importance measures model reliance, not causality.
- Correlated features can share or mask importance.
- Near-zero or negative values do not prove a feature is useless.
- Results are specific to this dataset and holdout split.
- The dataset contains synthetic observations.
- Independent external validation is still required.

Educational classification only; not a medical diagnostic tool.
