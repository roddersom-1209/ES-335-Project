# Contributions

**Team:** Rudra Mishra (25110276) and Rhinkar Rahul Shah (25110270)

## Who did what

| Area | What it involved | Who |
|---|---|---|
| Research question and project plan | Framing the question, choosing the dataset, writing the phase-by-phase brief and its rules | **Rudra** & **Rhinkar** |
| Data cleaning (`01_clean`, `src/features.py`) | Parsing the text columns, chip families and groups, dropping tablets, duplicates and phones without storage | **Rudra** |
| Manual web lookup (`data/manual/`) | Finding storage variants and confirming India launches and prices for phones without storage in their name | **Rhinkar** |
| Data quality checks | Data dictionary, India/China price check, flagging doubtful India prices | **Rhinkar** |
| Exploration (`02_explore`) | Price distribution, correlations, price by brand and chip | **Rudra** |
| Split and validation design (`src/split.py`) | Split by phone model, fixed cross-validation folds | **Rudra** |
| Baselines and OLS (`03_linear`, `03_linear_simple`) | Model A and Model B, coefficient tables, Apple-chip problem, multicollinearity (VIF, condition number) | **Rhinkar** |
| Regularised models and PCA (`04_regularised`, `04_regularised_simple`) | Ridge, lasso, elastic net, eigenvalue view, PCA of the specs | **Rhinkar** |
| Tree models (`05_trees`) | Decision trees, random forest, gradient boosting, AdaBoost, feature importance, interaction test | **Rudra** |
| Test-set evaluation (`06_evaluate`) | Single test evaluation, bootstrap intervals, paired comparisons, error analysis, robustness checks | **Rudra** |
| Brand premium and value analysis (`07_brand_value`) | Brand premiums and spec values with bootstrap intervals, per-phone value table, limitations | **Rudra** |
| Website data export and tests (`src/export_site.py`, `tests/`) | Exporting models, results and charts, checking the website's predictions against Python | **Rhinkar** |
| Website (`docs/`) | Page content and design, value explorer, price estimator, charts, dark mode, mobile layout | **Rhinkar** |
| README and documentation | README, this file, AI-use note | **Rudra** |

## Key decisions and who owns them

Each of us should be able to explain every decision. Write who mainly made or argued for each one.

| Decision | Why we made it | Owner |
|---|---|---|
| Predict **log price** instead of price | Spec effects act as percentages, and prices are right-skewed | **Rhinkar** |
| **Split by phone model**, not by row | Storage variants of one phone are near-duplicates. A random split would leak them into the test set | **Rhinkar** |
| Group Apple chips with **flagship Snapdragon** chips (`chip_group`) | Otherwise "Apple chip" and "Apple brand" are the same column and the brand premium cannot be estimated | **Rudra** |
| Group very small brands into **"Other"** | Avoid brand estimates based on a handful of phones | **Rhinkar** |
| **Drop phones without storage**, add back only confirmed India launches | Storage is a key price driver, and guessing it would bias the results | **Rudra** |
| Save the **cross-validation folds** in `split.csv` | Scores are identical on every machine and every scikit-learn version | **Rudra** |
| Use the **test set once** | An honest estimate of accuracy on unseen phones | **Rhinkar** |
| **Ridge** for brand premiums, **gradient boosting** for accuracy | Ridge gives one interpretable premium per brand, gradient boosting predicts best | **Rudra** |
| Add **PCA** and **AdaBoost** | Cover course topics. PCA explains how much the specs overlap, and AdaBoost is compared with gradient boosting | **Rudra** |
| Website: **value explorer** and **price estimator** | Let readers explore where the model works and fails, and see the hardware vs brand split | **Rhinkar** |

## Use of AI tools

See the "Use of AI tools" section of the [README](README.md).
