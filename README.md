# Explainable AutoML Customer-Churn MLOps Project

## Project overview

This repository contains the reproducibility, testing and CI/CD evidence for an MSc Data Science project comparing a manually engineered Random Forest pipeline with an Azure AutoML churn model.

The technical workflow covered:

- customer-level train, validation and test partitioning;
- manual Logistic Regression and Random Forest modelling;
- Azure AutoML comparison;
- same-test evaluation;
- bootstrap confidence intervals and paired error analysis;
- global and local SHAP explanations;
- profit-aware threshold sensitivity analysis;
- observed and simulated drift monitoring;
- automated model-quality checks.

## Final same-test performance

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

Azure AutoML produced the stronger overall accuracy, precision, F1, ROC-AUC and average precision. The Random Forest achieved higher recall and balanced accuracy, showing a different operational trade-off.

## Repository structure

```text
customer-churn-mlops/
├── src/                       Reusable validation and monitoring code
├── tests/                     Unit tests and model-quality gates
├── scripts/                   Repository evidence-validation script
├── notebooks/                 Final analytical notebooks
├── evidence/                  Selected dissertation evidence
├── .github/workflows/         GitHub Actions workflow
├── requirements.txt
├── model_card.md
├── config.json.example
└── README.md
```

## Automated quality gates

The workflow verifies that:

- required prediction columns exist;
- the target contains only 0 and 1;
- probabilities are complete and lie within 0 and 1;
- the final prediction file contains 94,669 test records;
- ROC-AUC is at least 0.90 for both models;
- average precision is at least 0.70 for both models;
- customer overlap is zero;
- PSI values are non-negative;
- the final comparison contains both models.

The workflow does not retrain the full 757,348-row model. It checks reusable code and frozen evidence so that CI remains fast and reproducible.

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install and test:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python scripts/validate_evidence.py
```

## CI/CD workflow

GitHub Actions runs on pushes and pull requests to `main`. It installs Python 3.10, runs the test suite, validates the frozen evidence and uploads the JUnit report and CI summary as an artefact.

## Data governance

Raw customer-level datasets, saved production models, Azure configuration files and secrets are excluded from version control. Only anonymised aggregate evidence and prediction outputs required for reproducibility are included.

## Important limitations

- Profit estimates are scenario-based sensitivity results, not audited company profit forecasts.
- Profit thresholds were selected using the available final test predictions and require future validation.
- The drift stress test was synthetic and severe; it was not a forecast of real customer behaviour.
- The local drift stress test rescored the Random Forest pipeline. Azure AutoML should be monitored through its isolated scoring environment or deployed endpoint.


The primary AutoML explanation uses model-agnostic permutation SHAP for the complete Voting Ensemble.

