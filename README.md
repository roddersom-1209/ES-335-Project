# What sets a smartphone's price in India: specs or brand?

An ES 335 (Machine Learning) course project at IIT Gandhinagar by **Rudra Mishra (25110276)** and **Rhinkar Rahul Shah (25110270)**.

**Live website:** <https://roddersom-1209.github.io/ES-335-Project/>  
**Repository:** <https://github.com/roddersom-1209/ES-335-Project>

We predict the **Indian launch price** of 702 smartphones from their listed specs, chip and brand, and measure how much extra each brand charges for the same hardware (the **brand premium**).

## The question

> How accurately can a smartphone's Indian launch price be predicted from its listed specifications, chip and brand, and how much of the price is explained by brand rather than hardware?

- **Target:** the launch price in India, in rupees (₹). The models predict the natural log of the price, because spec effects act as percentages and prices are right-skewed.
- **One row** is one phone variant: a specific RAM/storage configuration of a model (iPhone 16 128GB and iPhone 16 256GB are two rows).
- **Prediction time:** when a phone's specs, chip, brand and launch year are known but its Indian price is not. No input is derived from the price or from other countries' prices.

## Key findings

1. **Price can be predicted well.** On 144 phones the models never saw, gradient boosting reaches R² **0.932** on log price and a median error of **12.4%** (95% interval 9.6%–14.3%). 72% of phones are priced within ±20%.
2. **Hardware explains most of the price, and brand adds a clear extra.** Specs and chip alone reach R² 0.785, adding brand (ridge) gives 0.849, and brand alone only 0.512.
3. **Brand premiums (same specs and chip group, vs an average brand):** Apple +97% (relative to a phone with a flagship chip), Google +58%, Samsung +20%, down to Infinix −34%.
4. **What specs are worth:** doubling RAM and storage together goes with about **+65%** (+50% to +79%). Camera megapixels add nothing measurable.
5. **Tree models beat linear ones:** gradient boosting has a median error of 12.4%, against 21.5% for ridge. The gap is clearly real (95% interval 2.8 to 14.6 percentage points), mainly because brands price their budget and flagship lines differently. Gradient boosting, the random forest and AdaBoost cannot be separated from each other.

## Data

- **Source:** Abdul Malik, *Mobiles Dataset (2025)*, Kaggle, version 1 (updated 18 February 2025), Apache 2.0 licence. <https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025>, DOI [10.34740/kaggle/dsv/10780255](https://doi.org/10.34740/kaggle/dsv/10780255). A copy is in `data/raw/`.
- **Size:** 930 rows in the raw file, 702 phones after cleaning (15 brands, 9 chip groups). Split by phone model into 558 training and 144 test phones.
- **Quality issues:**
  - Storage size is written only in the model name, so phones without it were dropped. 7 were recovered by a manual web lookup (sources in `data/manual/`).
  - The dataset has no per-row price sources.
  - 24 phones have a doubtful India price (Huawei phones not sold in India, and unusual India/China price ratios). They are flagged, not dropped.

## Methods in brief

- **Preprocessing:** log price as the target, log₂ RAM and storage, processor names grouped into chip groups (Apple chips grouped with flagship Snapdragon chips), small brands grouped as "Other", one-hot encoding, standardised specs for the penalised linear models. All of it sits inside scikit-learn pipelines, fitted on training data only.
- **Models:** two baselines (overall mean, brand mean), OLS (specs + chip, and + brand), ridge, lasso, elastic net, decision trees, random forest, gradient boosting and AdaBoost. PCA is used to study how much the specs overlap.
- **Evaluation:** A split by phone model (all storage variants on the same side), 5-fold grouped cross-validation on the training set to choose every setting, one evaluation on the locked test set, and 1,000 bootstrap resamples for 95% intervals.
- **Metrics:** R² on log price, median absolute % error of the price, mean absolute error in ₹.

## Results on the test set

| Model | Test R² (95% interval) | Test median % error (95% interval) | CV R² | CV median % error |
|---|---|---|---|---|
| Gradient boosting | 0.932 (0.897 to 0.955) | 12.4% (9.6% to 14.3%) | 0.914 | 12.7% |
| Random forest | 0.925 (0.884 to 0.951) | 12.7% (10.3% to 15.8%) | 0.909 | 13.2% |
| AdaBoost | 0.926 (0.887 to 0.951) | 13.3% (11.6% to 16.1%) | 0.916 | 11.6% |
| Decision tree (best depth) | 0.846 (0.766 to 0.896) | 16.7% (12.0% to 20.0%) | 0.832 | 16.1% |
| Lasso | 0.855 (0.793 to 0.899) | 20.9% (15.8% to 25.9%) | 0.848 | 18.0% |
| Elastic net | 0.854 (0.792 to 0.898) | 21.0% (15.4% to 26.5%) | 0.848 | 18.1% |
| Ridge | 0.849 (0.785 to 0.896) | 21.5% (15.9% to 26.3%) | 0.847 | 18.3% |
| Decision tree (depth 3) | 0.795 (0.713 to 0.852) | 21.6% (19.8% to 28.1%) | 0.675 | 26.4% |
| Model B: specs + chip + brand | 0.851 (0.789 to 0.896) | 21.6% (15.4% to 27.1%) | 0.845 | 18.2% |
| Model A: specs + chip | 0.785 (0.683 to 0.853) | 23.7% (18.3% to 27.8%) | 0.773 | 23.3% |
| Baseline 2: brand mean | 0.512 (0.323 to 0.645) | 33.5% (28.4% to 42.6%) | 0.366 | 40.3% |
| Baseline 1: overall mean | −0.014 (−0.137 to 0.000) | 58.7% (50.4% to 67.7%) | −0.006 | 59.1% |

Brand premiums (ridge, same specs and chip group, vs an average brand, 95% bootstrap intervals):

| Brand | Premium | 95% interval |
|---|---|---|
| Apple | +97% | +68% to +133% |
| Other (Sony, Huawei) | +82% | +46% to +110% |
| Google | +58% | +34% to +83% |
| Samsung | +20% | +5% to +39% |
| Xiaomi | +8% | −6% to +24% |
| Vivo | −1% | −14% to +12% |
| Oppo | −2% | −16% to +15% |
| Nokia | −10% | −16% to −2% |
| Tecno | −12% | −22% to +0% |
| Motorola | −15% | −24% to −5% |
| OnePlus | −20% | −26% to −12% |
| POCO | −26% | −33% to −18% |
| Realme | −27% | −34% to −19% |
| Lenovo | −28% | −34% to −19% |
| Infinix | −34% | −41% to −26% |

The Google and "Other" (Sony, Huawei) premiums partly include chip value, because the Tensor and Kirin chips are used by one brand each.

## Repository structure

```
data/raw/            the Kaggle CSV (never edited)
data/manual/         manual web lookup of missing storage, with sources
data/processed/      phones_clean.csv and split.csv (fixed train/test split and CV folds)
notebooks/           the analysis, run in order (see below)
src/                 features.py (parsing), split.py (split and folds), evaluate.py, export_site.py
results/             metrics (CSV), chosen settings (best_params.json), figures (PNG)
models/              fitted ridge, Model A and gradient boosting (used by the website test)
tests/               test_export.py
docs/                the website (served by GitHub Pages)
```

## Installation

Python 3.14 was used. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

The cleaned data and the split are already in `data/processed/`, so you can start at any notebook. To rebuild everything from the raw file, run in this order (from the `notebooks/` folder in Jupyter, or with `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`):

| Step | What it does |
|---|---|
| `01_clean.ipynb` | Problem statement, data source and dictionary, parsing, cleaning, India-price check. Writes `phones_clean.csv`. |
| `python src/split.py` | Fixed train/test split by phone model, and the 5 cross-validation folds. Writes `split.csv`. |
| `02_explore.ipynb` | Exploration: price distribution, correlations, price by brand and chip. |
| `03_linear_simple.ipynb` | Baselines and OLS (Model A: specs + chip, Model B: + brand), multicollinearity. |
| `04_regularised_simple.ipynb` | Ridge, lasso, elastic net, and PCA of the specs. |
| `05_trees.ipynb` | Decision trees, random forest, gradient boosting, AdaBoost, feature importance. Writes `best_params.json`. |
| `06_evaluate.ipynb` | The single evaluation on the test set, bootstrap intervals, error analysis. |
| `07_brand_value.ipynb` | Brand premiums, spec values and the per-phone value table. |
| `python src/export_site.py` | Writes the website's data and charts into `docs/`. |

`03_linear.ipynb` and `04_regularised.ipynb` are earlier versions of notebooks 03 and 04 with the same results. The `_simple` versions are the main ones.

## Results and figures

- **Metrics:** `results/cv_phase4.csv`, `cv_phase5.csv` and `cv_phase6.csv` (cross-validation), `test_scores.csv` and `test_bootstrap.csv` (test set).
- **Brand and spec analysis:** `results/brand_premiums.csv`, `spec_values.csv` and `phone_values.csv`.
- **Figures:** `results/fig*.png`.

## Tests

`tests/test_export.py` checks that the website computes exactly the same predictions as the Python models: ridge and Model A from `docs/model.json`, and gradient boosting by walking its trees in `docs/boosting.json`.

```bash
python tests/test_export.py
```

## Viewing the website locally

The page loads its data from JSON files, so it must be served, not opened as a file:

```bash
python3 -m http.server --directory docs
```

Then open <http://localhost:8000>.

## Limitations

Launch prices come from one dataset without per-row sources. Quality factors such as camera quality, display type, materials and software updates are not in the data. Phones without a storage size in their name were dropped, and small brands have wide intervals. Brand premiums depend on the chip grouping (Apple chips grouped with flagship Snapdragon chips). The results are associations in launch prices, not causes, and the test set is small (144 phones). See the website's Limitations section for details.

## Use of AI tools

We used **Claude** (Anthropic), an AI assistant, throughout this project, as permitted by the course. It was used to help us with writing the code and build up the website for this project.

## Contributions

See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).
