# NK_Securities_Hackathon
MY kaggle notebook code of NKSR Volatility Curve Prediction Hackathon.\
Score based on MSE , lesser the MSE score better the performance.\
Private Score : 0.000002153\
Public Score : 0.000001872\
Final Rank : 32\
Kaggle link for Problem Statement : https://www.kaggle.com/competitions/nk-iv-prediction\
Implied Volatility Imputation & Extrapolation
==============================================================

Author: Sapaharam Vishnu Singh , 220108053 , IIT GUWAHATI

Objective:
----------

To impute missing implied volatility (IV) values in the NIFTY options dataset and extrapolate unavailable strike IVs using a polynomial fitting model.

Pipeline Summary:
-----------------

1. Data Loading:

   - Loaded training and test datasets from `.parquet` files.
   - Used `sample_submission.csv` to identify all target IV columns 

2. Feature Filtering:

   - Identified numerical features common to both train and test.
   - Removed sparse features (more than 50% zeros). They are not meaningful enough.
   - Combined valid features with IV columns to form the imputation feature set.

3. Missing Value Imputation:

   - Used `IterativeImputer` from `scikit-learn`, with `XGBRegressor` as the estimator.
   - XGBoost settings: 200 trees, max depth 7, learning rate 0.08.
   - If GPU is available, `gpu_hist` and `gpu_predictor` are enabled for acceleration.

4. Extrapolation:

   - Applied a 5-degree polynomial fit (using `PolynomialFeatures`) on observed strike-IV pairs.
   - Fitted a `BayesianRidge` regression for each row, separately for `call_iv_*` and `put_iv_*`.
   - Used the fitted model to extrapolate IVs for missing strikes in test-only columns.

5. Submission Generation:

   - Final imputed IVs are inserted into the `sample_submission.csv` format.
   - Output saved as `imputed_submission.csv`.

Result:
-------

A complete submission with all missing IVs imputed and extrapolated, suitable for direct evaluation on the competition leaderboard.

Note:
-----

- Model is reproducible using `random_state=42`.
- Requires GPU-compatible XGBoost for optimal performance, but works on CPU by changing `tree_method` to `"hist"`.
-Please ensure all file paths are correct to avoid unnecessary errors.

