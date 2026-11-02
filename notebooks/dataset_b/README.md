# Dataset B Notebook Index

This directory separates historical development evidence from the current
reporting path. The migration was path-only: notebook cells, saved outputs,
execution counts, metadata, and error payloads were not edited or executed.

`final` means the notebook belongs to the current reporting contract or reads
authoritative frozen evidence. It does not mean the notebook was rerun or that
its outputs were newly validated during the handover refactor.

## Historical and development notebooks

| Path | Role | Authority |
|---|---|---|
| `v1/01_linear_model_v1.ipynb` | Logistic Regression historical experiment | Historical evidence only |
| `v1/02_random_forest_v1.ipynb` | Random Forest historical experiment | Historical evidence only |
| `v1/03_xgboost_v1.ipynb` | XGBoost historical experiment | Historical evidence only |
| `v1/05_realmlp_v1.ipynb` | RealMLP historical experiment | Historical evidence only |
| `v1/colab_env.ipynb` | Historical environment setup | Runtime reference only |
| `v1/eda/01_eda_search.ipynb` | Early EDA/search | Historical evidence |
| `v1/eda/04_feature_selection.ipynb` | Feature-selection evidence | Governance evidence |
| `v1/eda/05_feature_selection.ipynb` | Feature reconciliation evidence | Governance evidence |

The final EDA filename corrects the former spelling
`05_feature_selction.ipynb`; its bytes were not changed.

## Current reporting notebooks

### Models

| Path | Role |
|---|---|
| `final/models/01_linear_model_final.ipynb` | Logistic Regression final reporting |
| `final/models/02_random_forest_final.ipynb` | Random Forest final reporting |
| `final/models/03_xgboost_final.ipynb` | XGBoost final reporting |
| `final/models/04_plain_mlp_final.ipynb` | PlainMLP final reporting |
| `final/models/05_realmlp_final.ipynb` | RealMLP final reporting |
| `final/models/06_tabmmini_final.ipynb` | TabMmini final reporting |

### Research questions

| Path | Role |
|---|---|
| `final/research_questions/07_model_total_review.ipynb` | RQ1 cross-model review |
| `final/research_questions/08_rq2_gsv_sl_results.ipynb` | RQ2 explanation results |
| `final/research_questions/09_rq3_fairness_results.ipynb` | RQ3 fairness results |
| `final/research_questions/11_rq4a_equalized_odds_results.ipynb` | RQ4a results |
| `final/research_questions/12_rq4b_adversarial_results.ipynb` | RQ4b adversarial results |
| `final/research_questions/13_rq4b_applicant_sex_supplementary_results.ipynb` | RQ4b supplementary applicant-sex results |

### Reporting

| Path | Role |
|---|---|
| `final/reporting/10_publication_figures.ipynb` | Publication figures |
| `final/reporting/14_research_report.ipynb` | Research report assembly |

## Lineage and path policy

- Use the paths above for current navigation and tests that open notebooks.
- Historical artifacts, frozen configuration, generated reports, and notebook
  cell text may retain the pre-migration path as recorded lineage. Do not edit
  those frozen records merely to make their path current.
- Moving a notebook does not change its SHA-256. Notebook 12 and Notebook 13
  already differed from hashes recorded in historical STATUS evidence before
  this migration; T03 preserves their current bytes and does not resolve or
  conceal that pre-existing drift.
- Do not execute, clear, normalize, re-save, or convert notebooks as part of a
  path migration.
