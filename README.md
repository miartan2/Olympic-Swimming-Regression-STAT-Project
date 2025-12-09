# Olympic Swimming Performance Prediction
This project compares several regression techniques to predict Olympic swimming results, evaluating which modeling approach best captures variation in race times.

# Dataset
This project uses publicly available Olympic swimming results compiled by [Score Network](https://data.scorenetwork.org/swimming/olympic_swimming.html). The dataset inclues final times, swimmer metadata, and event details from 1924 to 2020 for 607 entries.
- Features: Year, Distance, Stroke, Gender, Team
- Target: Final time (in seconds)

# Methods
- Linear regression
- Polynomial regression (degree = 2)
- Spline regression (Year)
- Ridge regression
- Lasso regression

# Preprocessing
- Numerical features: standard scaling, polynomial expansion
- Categorical features: one-hot encoding
- Train/test split: 80/20

# Insights
- Polynomail regression (RMSE = 0.889; R^2 = 0.936) improved predictive accuracy over linear models, suggesting meaningful nonlinear interactions between features
- Regularization (ridge/lasso) did not outperform the simpler linear model, likely due to limited feature dimensionality
- Spline regression using only Year performed poorly, indicating that temporal strends alone cannot explain performance variation
- Residual diagnostics showed mild nonlinearity and slight deviations from normality

