# GitHub Setup Guide

1. Extract this ZIP.
2. Create an empty GitHub repository named `customer-churn-mlops`.
3. Open a terminal inside the extracted folder.
4. Run:

```bash
git init
git add .
git commit -m "Add churn MLOps quality pipeline"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

5. Open the repository's **Actions** tab.
6. Select **Model quality checks**.
7. Confirm that the workflow has a green check mark.
8. Open the completed run and download the `model-quality-evidence` artefact.
9. Capture screenshots of:
   - repository structure;
   - workflow YAML;
   - passed tests;
   - green workflow run;
   - uploaded artefact.

Do not commit Azure credentials, raw customer data, `config.json`, `.env`, saved `.pkl` files or saved `.joblib` files.
