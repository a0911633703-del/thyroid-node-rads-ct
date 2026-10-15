# Thyroid cervical lymph node CT and Node-RADS analysis

Analysis code, model coefficients and saved ridge logistic regression pipelines for manuscript HELIYON-D-26-04277 on cervical lymph node assessment in thyroid cancer.

Download **Code and README.zip** from this repository and extract it. The commands below are run from the extracted `Code and README` directory.

The de-identified measurements and held-out predictions are provided separately as **Supplementary Data.xlsx** with the journal submission. This repository contains code and fitted parameters. Download the workbook separately and keep it outside the repository.

## Files

```text
analysis_source/      11 analysis and plotting source scripts
parameters/           Coefficients, feature definitions and selected settings
saved_models/         30 fitted ridge logistic regression pipelines
configuration/        Primary and revision-added analysis settings
environment/          Recorded software versions and dependency snapshot
run_saved_prediction.py
predict_from_parameters.py
import_supplementary_workbook.py
```

The 30 saved pipelines consist of median imputation, standard scaling and logistic regression. For each of the five model specifications, five outer-fold pipelines and one full-development pipeline are supplied. Their stored arrays contain preprocessing statistics, coefficients and class labels.

| Saved model name | Manuscript model | Predictors |
| --- | --- | ---: |
| NodeRADS | Node-RADS-only model | 1 |
| Imaging | Quantitative CT model | 14 |
| Combined | Full combined model | 15 |
| RoutineCT | Conventional CT model | 4 |
| NodeRADS_RoutineCT | Conventional combined model | 5 |

## Calculate probabilities

The prediction example requires Python and openpyxl. Install openpyxl in your environment, then run:

```text
python run_saved_prediction.py --workbook "/path/to/Supplementary Data.xlsx" --output "/path/to/new outputs/probabilities.csv"
```

The script reads the `Nodes` sheet and selects its 248 primary-cohort nodes. It applies the full-development coefficients in `parameters/model_parameters.json`. Missing predictors use the saved imputation medians. A new output filename is required for each run.

To use the equivalent original-unit formula, add `--original-units`. To supply another predictor CSV, use `--input` in place of `--workbook`. The required column names and units are given in `parameters/variable_definitions.csv`.

These probabilities describe the full-development fits. The internal-validation probabilities, fold-specific thresholds and classifications are stored separately in the workbook's `OOF predictions` sheet.

## Import the workbook into an analysis workspace

```text
python import_supplementary_workbook.py --workbook "/path/to/Supplementary Data.xlsx" --workspace "/path/to/Analysis Workspace"
```

This creates a new directory, exports the measurements and saved OOF predictions to the historical analysis layout, and copies the analysis sources and saved models. It preserves the workbook's node order, patient groups and outer-fold assignments. Existing workspaces are protected against overwrite. Importing performs no model fitting or bootstrap resampling.

The workbook contains three sheets:

| Sheet | Contents |
| --- | --- |
| Nodes | 254 recorded nodes, coded identifiers, outcome, Node-RADS, 14 CT measurements, two reference-tissue CT values, cohort flags and outer fold |
| OOF predictions | 1,240 fixed held-out predictions: 248 primary-cohort nodes across five models, with probabilities, thresholds and classifications |
| Data dictionary | Field definitions, measurement units, cohort definitions and score interpretation |

The primary cohort includes 248 nodes from 130 recorded patient groups, with 84 malignant and 164 benign nodes. The remaining source nodes support the reported size-criterion sensitivity comparisons. Nodal outcome was assigned from concordant cervical-level pathology. The two reference measurements are CT attenuation values in HU.

## Analysis sources

| Script | Role |
| --- | --- |
| training.py | Nested patient-grouped ridge models, sensitivity cohorts, 11 algorithm comparisons and primary SHAP values |
| evaluate_results.py | Fixed-OOF performance, calibration, decision curves and patient-group bootstrap intervals |
| run_incremental_analysis.py | Conventional CT control models and paired comparisons |
| raw_noderads_benchmark.py | Original ordinal Node-RADS benchmark |
| reference_shap_comparison.py | SHAP comparison after adding reference CT attenuation |
| make_regression_forests.py | GEE associations and fixed-C ridge coefficient bootstrap |
| make_extended_validation_figures.py | Algorithm, sensitivity, confusion and patient-level summaries |
| make_algorithm_probability_figures.py | Calibration and decision curves for probability-output algorithms |
| make_extended_shap_figures.py | Additional SHAP displays |
| make_extended_descriptive_figures.py | Distributions, correlations, nomogram and tuning displays |
| make_figures.py | Shared plotting functions and base figures |

The source scripts retain the original statistical calculations, random seed `20260926`, five grouped outer folds, three grouped inner folds, and training-fold imputation, scaling, penalty selection and threshold selection. Main-model tuning uses negative log loss. Algorithm comparison uses ROC AUC. SVM uses its decision function.

For the original three ridge models, the imported workspace supports:

```text
python "/path/to/Analysis Workspace/code/training.py" --stage primary
```

Additional stages are `sensitivity` and `exploratory`. The latter includes 11 algorithms and requires the full scientific environment. The conventional CT controls were added during revision using the same primary nodes and outer-fold allocation.

Some plotting scripts consume generated intermediate files. Run their upstream analysis first and review the input filenames in each source. Historical figure numbers may differ from the final manuscript composition. The supplement workbook retains the measurements and five main-model OOF records, while intermediate plot coordinates, bootstrap draws and logs are generated by analysis. Complete retraining and regeneration of every final figure have not been rerun for this repository release.

Performance intervals use 2,000 patient-group bootstrap resamples of fixed OOF predictions. Models, tuning and thresholds remain fixed within those resamples. Calibration parameters are re-estimated. The coefficient bootstrap separately refits preprocessing and regression coefficients with C fixed at 1.

## Environment and release checks

The original analysis environment was Python 3.14.3 on Windows. `environment/requirements-analysis-observed.txt` records exact observed dependency versions. This snapshot has not been tested as a fresh cross-platform installation. Use a separate environment for scientific analysis.

Host-specific file paths, internal clinical-data preparation records and packaging logs were excluded from this public release. Source edits for distribution are limited to file locations and the workbook import interface. Scientific settings are retained in `configuration/analysis_settings.json` and `configuration/incremental_analysis_settings.json`. These settings describe the analysis and are separate from prospective study registration.

Parameter-based probability calculations and the workbook import are checked locally against the supplied data. All five full-development pipelines and 25 outer-fold pipelines are checked against their stored parameters or corresponding OOF predictions. `release_validation.json` records the checks completed for this release. A fresh full training and bootstrap run is outside those release checks.
