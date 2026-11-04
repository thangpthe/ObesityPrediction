# UCI dataset assessment — ObesityPrediction

> **Inspected:** 2026-11-04  
> **Uploaded archive:** `estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip`  
> **Extracted CSV:** `ObesityDataSet_raw_and_data_sinthetic.csv`  
> **Decision:** Accept as a documented educational multiclass dataset candidate; do not describe it as a fully raw or clinically representative dataset.

## 1. Provenance

The uploaded archive matches the official UCI Machine Learning Repository dataset **Estimation of Obesity Levels Based on Eating Habits and Physical Condition**.

- Official record: https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition
- DOI: https://doi.org/10.24432/C5H31Z
- Original article: Palechor & de la Hoz Manotas (2019), *Data in Brief*, DOI https://doi.org/10.1016/j.dib.2019.104344
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution is required.
- Study population/context: individuals from Mexico, Peru and Colombia.

The file is called `raw_and_data_sinthetic` (original spelling in the source). It is **not a pure raw-only file**: the source article reports 485 collected records and a final dataset expanded to 2,111 records, with 77% synthetic records generated using Weka/SMOTE and 23% collected through a web survey. The CSV does not provide a per-row flag identifying which records are original versus synthetic.

## 2. File integrity

| Item | Value |
|---|---|
| Archive size | 57,676 bytes |
| Extracted CSV size | 263,646 bytes |
| CSV rows | 2,111 |
| CSV columns | 17 (16 features + 1 target) |
| Missing cells | 0 |
| Exact duplicate rows | 24 |
| Unique rows after exact deduplication | 2,087 |
| CSV SHA-256 | `28b6991062855d3d15642e38b1aa9d1b32fb7abbb5f5b096f1868cd706154b` |
| ZIP SHA-256 | `9cf2ee5e3c3139815c059217a832ff3c0a4ec07bc55b80245f1b94bb3254b297` |

The duplicate rows are exact duplicates. They occur in Insufficient_Weight (8 duplicate-row instances), Normal_Weight (10) and Overweight_Level_I (15); the sum of these class-level counts is not the total duplicate-row count because duplicate groups may contain more than two copies. Deduplicate exact rows before splitting, and record the removal count.

## 3. Schema

Features:
- Demographic/anthropometric: `Gender`, `Age`, `Height`, `Weight`.
- Family history: `family_history_with_overweight`.
- Eating habits: `FAVC`, `FCVC`, `NCP`, `CAEC`, `CH2O`, `CALC`.
- Lifestyle/activity: `SMOKE`, `SCC`, `FAF`, `TUE`, `MTRANS`.

Target: `NObeyesdad`, with seven categories:
`Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`.

All columns have zero missing values. The categorical columns are stored as strings; numeric survey scales such as FCVC/NCP/CH2O/FAF/TUE are numeric and should be treated according to the original variable descriptions, not automatically assumed to be continuous measurements.

## 4. Target distribution

| Target | Rows |
|---|---:|
| Insufficient_Weight | 272 |
| Normal_Weight | 287 |
| Overweight_Level_I | 290 |
| Overweight_Level_II | 290 |
| Obesity_Type_I | 351 |
| Obesity_Type_II | 297 |
| Obesity_Type_III | 324 |
| **Total** | **2,111** |

The classes are relatively balanced compared with a natural population sample. This is consistent with the source's synthetic balancing process; it must not be presented as a representative prevalence distribution.

## 5. Target definition and leakage risk

The original article describes calculating BMI as:

`BMI = Weight / Height²`

and assigning weight-status categories using BMI thresholds. The released CSV does not include a BMI column, but it does include both Height and Weight. Therefore, a model trained with Height and Weight can reconstruct a major part of the rule used to create the target. This is target leakage / a shortcut risk if the intended product is supposed to estimate obesity level from lifestyle and demographic questionnaire answers.

**Recommended primary experiment:** retain the original seven-class `NObeyesdad` target, but exclude `Height` and `Weight` from model inputs. Derive BMI from them only for audit/EDA, never as a predictor in this experiment. Explicitly state that the labels themselves are BMI-derived, so this is an educational prediction of the dataset's BMI-defined category from survey/lifestyle variables, not independent clinical diagnosis.

An optional comparison experiment may train a separate anthropometric baseline using Height and Weight. Label it clearly as a BMI-rule reconstruction baseline, not as evidence of clinically useful prediction.

## 6. Evaluation limitations

- The dataset combines collected and SMOTE-generated records, but row-level provenance is unavailable.
- A random stratified split can put synthetic neighbors and related source observations into both train and test partitions. Exact duplicate removal does not solve this broader issue.
- There is no reliable way to isolate a held-out test set containing only the 23% collected observations from this CSV alone.
- Do not claim external validity, clinical performance, population prevalence, or deployment readiness from this dataset.
- Report macro-F1, per-class precision/recall/F1, confusion matrix and stratified cross-validation. Include class-wise results, not accuracy alone.
- If a genuinely independent assessment is required, obtain a separate, consented, appropriately labeled cohort with documented collection and target measurement protocol.

## 7. Comparison with the repository's legacy binary dataset

The legacy `dataset/obesity_prediction_dataset_final.csv` has 17,698 rows, 9 columns and a binary `Obesity` target, including a precomputed BMI field. The uploaded UCI file has 2,111 rows, 17 columns and the original seven-class `NObeyesdad` target, with no BMI field. These are not interchangeable files and should not be concatenated or treated as the same raw source without a documented transformation lineage.

The legacy binary file's source and label mapping remain unverified. Keep it quarantined as a legacy experiment until its original source files and label-generation code are recovered.

## 8. Repository placement

After downloading the uploaded ZIP locally, place the original archive under:

`data/raw/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip`

Extract the CSV to:

`data/raw/ObesityDataSet_raw_and_data_sinthetic.csv`

Keep both source archive and extracted CSV unchanged. Record the SHA-256 checksums above. Add a source citation and CC BY 4.0 attribution in `data/raw/README.md`. The raw archive is not committed by this documentation-only change; use the local Git commands in the main response to add it to the public repository.

## 9. Recommendation

Use this UCI dataset as the new, documented educational baseline for a **seven-class supervised classification** project, subject to the limitations above. Do not silently convert it to binary or merge it with the legacy 17,698-row dataset. Keep the original target labels, exclude Height/Weight from the primary lifestyle-only model, and make the synthetic-data limitation prominent in the README and CV description.
