# Model Card: Customer-Churn Prediction

## Model versions

- Manual model: Random Forest pipeline
- Automated model: Azure AutoML Voting Ensemble
- Decision threshold used for the final statistical comparison: 0.50
- Random seed: 42

## Intended use

The models estimate customer-churn risk to support prioritisation of retention activity. Predictions are advisory and should be combined with business rules, campaign capacity and human oversight.

## Out-of-scope use

The models should not be used to make decisions about credit, employment, insurance, essential services or other high-impact individual outcomes.

## Data

- Training records: 757,348
- Validation records: 94,669
- Test records: 94,669
- Predictors: 29
- Test churn rate: approximately 8.99%
- Customer overlap between training, validation and test: zero

## Final test performance

| Metric | Random Forest | Azure AutoML Voting Ensemble |
|---|---:|---:|
| Accuracy | 0.9464 | 0.9701 |
| Balanced accuracy | 0.9013 | 0.8783 |
| Precision for churn | 0.6568 | 0.8852 |
| Recall for churn | 0.8464 | 0.7665 |
| F1 for churn | 0.7396 | 0.8216 |
| ROC-AUC | 0.9654 | 0.9759 |
| Average precision | 0.8426 | 0.8922 |
| Log loss | 0.2303 | 0.0916 |

## Explainability

Global and local SHAP analyses were produced for the manual and AutoML workflows. Seven of the top ten grouped features overlapped between the two explanation sets. Important themes included payment method, transaction history, renewal behaviour, recency and activity.

## Profit-aware use

Under the central illustrative scenario, Azure AutoML achieved the highest estimated net profit at a probability threshold of 0.33. These values depend on assumed customer value, intervention effectiveness and campaign cost. They are sensitivity results rather than guaranteed profit.

## Drift monitoring

Observed train, validation and test feature distributions were stable. A severe synthetic covariate-drift stress test triggered feature, prediction and performance alerts. Monitoring thresholds used in the project were:

- PSI below 0.10: stable
- PSI from 0.10 to below 0.25: warning
- PSI of 0.25 or above: significant

## Retraining and review triggers

Review or retraining should be considered when one or more of the following occurs:

- important-feature PSI reaches 0.25;
- prediction PSI reaches 0.25;
- ROC-AUC, average precision or churn F1 declines by at least 0.03;
- churn recall declines by at least 0.05;
- predicted churn volume changes by at least 20%;
- data schema, missingness or category levels change unexpectedly.

## Risks and limitations

- Historical behaviour may not represent future populations.
- A fixed 0.50 threshold may not be operationally optimal.
- Synthetic drift does not reproduce concept drift.
- Profit assumptions require validation with real commercial data.
- Explanations describe model behaviour and do not establish causality.
- Performance should be reviewed across relevant demographic and customer groups before deployment.

## Human oversight

Campaign owners should review the selected threshold, capacity, false-positive cost and customer-treatment policy before using predictions operationally.


The primary AutoML explanation uses model-agnostic permutation SHAP for the complete Voting Ensemble. Earlier XGBoost SHAP evidence is supplementary only.
