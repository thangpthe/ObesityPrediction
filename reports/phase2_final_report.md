# Phase 2 – Model Development, Evaluation and Analysis

## 1. Overview

Phase 2 focuses on developing, evaluating, and analyzing machine learning models for obesity level classification using the UCI Estimation of Obesity Levels Based on Eating Habits and Physical Condition dataset.

The main objectives are:

- Establish a reproducible machine learning training pipeline.
- Compare multiple classification algorithms.
- Evaluate model performance using stratified cross-validation and a holdout set.
- Analyze feature importance.
- Investigate model performance across gender subgroups.
- Examine the contribution of the Gender feature.
- Optimize Random Forest hyperparameters.

The task is treated as an educational multi-class classification problem, not a clinical diagnostic application.

## 2. Dataset and Preprocessing

The original dataset contains 2,111 records and 17 columns, including the target variable `NObeyesdad`.

After removing 24 exact duplicate records, 2,087 records remained.

The target variable contains seven obesity-related categories:

- Insufficient_Weight
- Normal_Weight
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III
- Overweight_Level_I
- Overweight_Level_II

The primary model uses 14 input features. `Height` and `Weight` are excluded because they are directly related to BMI-based target construction and could create an anthropometric shortcut.

Preprocessing includes:

- Numerical and categorical feature separation.
- Missing-value imputation.
- Numerical feature scaling.
- Categorical one-hot encoding.
- Stratified train-test splitting.
- Preprocessing fitted inside the machine learning pipeline to prevent data leakage.

The dataset was divided into:

- Training set: 1,669 samples.
- Holdout set: 418 samples.

Five-fold StratifiedKFold cross-validation was performed on the training set.

## 3. Baseline Model Development

Four classification approaches were compared:

| Model               |     CV Macro-F1 |
| ------------------- | --------------: |
| Random Forest       | 0.8384 ± 0.0132 |
| Decision Tree       | 0.7221 ± 0.0265 |
| Logistic Regression | 0.5730 ± 0.0312 |
| Dummy Classifier    | 0.0412 ± 0.0002 |

Random Forest achieved the highest cross-validation Macro-F1 and was selected as the primary model.

### Holdout Evaluation

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 0.8421 |
| Balanced Accuracy | 0.8394 |
| Macro Precision   | 0.8382 |
| Macro Recall      | 0.8394 |
| Macro-F1          | 0.8379 |
| Weighted-F1       | 0.8415 |

The model achieved relatively balanced performance across the seven target categories, although some classes remained more difficult to classify.

The weakest class-level F1 scores included:

- Overweight_Level_II: 0.71
- Normal_Weight: 0.74
- Overweight_Level_I: 0.77

Obesity_Type_III achieved the highest F1 score of 0.99.

## 4. Feature Importance Analysis

Permutation importance was calculated on the holdout set using 30 repetitions and Macro-F1 as the scoring metric.

The most influential features according to model performance degradation were:

| Feature                        | Mean Importance |
| ------------------------------ | --------------: |
| Gender                         |         0.26261 |
| Age                            |         0.17435 |
| FCVC                           |         0.16828 |
| family_history_with_overweight |         0.16056 |
| CALC                           |         0.11080 |
| NCP                            |         0.07641 |
| CAEC                           |         0.06957 |
| TUE                            |         0.05088 |
| CH2O                           |         0.04486 |
| FAF                            |         0.04035 |

Gender had the highest permutation importance, followed by Age, FCVC, and family history of overweight.

These values represent the model's reliance on individual features. They do not establish causal relationships.

The low importance of SMOKE does not independently justify removing it from the model.

## 5. Gender Subgroup Evaluation

The holdout set was analyzed separately by gender.

| Gender | Samples | Accuracy | Macro-F1 |
| ------ | ------: | -------: | -------: |
| Female |     198 |   0.8990 |   0.7235 |
| Male   |     220 |   0.7909 |   0.6670 |

Female accuracy was higher than male accuracy in this holdout evaluation.

However, the subgroup Macro-F1 scores require careful interpretation because the target distributions differ substantially between groups.

For example:

- Female samples contained no Obesity_Type_II instances.
- Male samples contained only one Obesity_Type_III instance.

Consequently, some class-level metrics are based on absent or extremely small supports.

These results indicate a need for further subgroup investigation. They do not, by themselves, establish that the model is biased or fair.

## 6. Gender Ablation Study

An ablation experiment compared Random Forest models trained with and without the Gender feature.

Both experiments used the training partition and five-fold stratified cross-validation.

| Experiment     | Features |        Macro-F1 | Balanced Accuracy | Accuracy |
| -------------- | -------: | --------------: | ----------------: | -------: |
| With Gender    |       14 | 0.8384 ± 0.0132 |            0.8388 |   0.8424 |
| Without Gender |       13 | 0.8060 ± 0.0191 |            0.8066 |   0.8101 |

Including Gender increased the mean CV Macro-F1 by 0.0324.

This suggests that Gender provides predictive information in this dataset.

The result is not a causal analysis and does not establish that using Gender is appropriate in every deployment context.

The primary model retains Gender, while subgroup performance and fairness limitations remain important considerations.

## 7. Random Forest Hyperparameter Tuning

RandomizedSearchCV evaluated 30 hyperparameter configurations using five-fold StratifiedKFold cross-validation.

Macro-F1 was selected as the optimization metric.

The best configuration was:

| Parameter         | Selected Value |
| ----------------- | -------------- |
| n_estimators      | 400            |
| max_depth         | 30             |
| min_samples_split | 2              |
| min_samples_leaf  | 2              |
| max_features      | sqrt           |
| class_weight      | balanced       |

### Tuning Results

| Configuration | Mean CV Macro-F1 | Standard Deviation |
| ------------- | ---------------: | -----------------: |
| Baseline      |           0.8384 |             0.0132 |
| Best tuned    |           0.8384 |             0.0132 |

The tuning process did not produce a measurable improvement over the baseline.

Therefore, the original Random Forest configuration is retained as the primary model. The tuned model is preserved separately for reproducibility and comparison.

The previously examined holdout results are reported for transparency only and are not used for further model selection.

## 8. Limitations

Several limitations must be considered:

1. **Synthetic data:** The UCI dataset contains a substantial proportion of synthetic records. There is no row-level original/synthetic indicator available for separating these records reliably.
2. **Potential dependence:** Related records may appear across training and holdout partitions, potentially affecting generalization estimates.
3. **Target construction:** The target is BMI-related, while Height and Weight are excluded from the primary model to reduce direct anthropometric shortcuts.
4. **Model selection optimism:** Searching multiple configurations can lead to optimistic cross-validation estimates.
5. **Previously examined holdout:** The holdout set has been examined in multiple analyses and should not be treated as an untouched final evaluation set.
6. **Subgroup imbalance:** Gender subgroup target distributions are different, and some class-level supports are very small or absent.
7. **External validity:** The dataset is not a clinical validation cohort. Results cannot establish clinical usefulness or diagnostic reliability.

## 9. Conclusion

Phase 2 established a reproducible machine learning pipeline and selected Random Forest as the strongest model among the evaluated baseline algorithms.

The model achieved a holdout Macro-F1 of 0.8379 and a cross-validation Macro-F1 of 0.8384.

Feature importance and gender subgroup analyses provided additional insight into model behavior. The ablation experiment showed that Gender contributes predictive information in the current dataset.

Hyperparameter tuning did not improve upon the baseline configuration, supporting the decision to retain the original Random Forest model.

Further research should prioritize independent external validation, more robust subgroup evaluation, and careful assessment of potential data dependence before making claims about generalization or real-world applicability.

**Phase 2 status: Completed – baseline model development, evaluation, and exploratory analysis.**
