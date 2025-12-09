# -*- coding: utf-8 -*-
"""
Olympic Swimming Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from patsy import dmatrix
import statsmodels.api as sm


# Load Data
df = pd.read_csv("olympic_swimming.csv")

print(df.head())
print(df.info())

y = df["Results"]
X = df[["Year", "dist_m", "Stroke", "Gender", "Team"]]

num_features = ["Year", "dist_m"]
cat_features = ["Stroke", "Gender", "Team"]


# Preprocessing Pipelines
numeric_standard = Pipeline([
    ("scaler", StandardScaler())
])

numeric_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False))
])

categorical = Pipeline([
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

pre_basic = ColumnTransformer([
    ("num", numeric_standard, num_features),
    ("cat", categorical, cat_features)
])

pre_poly = ColumnTransformer([
    ("num", numeric_poly, num_features),
    ("cat", categorical, cat_features)
])


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Linear Regression
linreg = Pipeline([
    ("pre", pre_basic),
    ("reg", LinearRegression())
])

linreg.fit(X_train, y_train)
y_pred_lin = linreg.predict(X_test)

print("\nLinear Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lin)))
print("R^2:", r2_score(y_test, y_pred_lin))


# Polynomial Regression (better structure)
polyreg = Pipeline([
    ("pre", pre_poly),
    ("reg", LinearRegression())
])

polyreg.fit(X_train, y_train)
y_pred_poly = polyreg.predict(X_test)

print("\nPolynomial Regression (degree=2)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))
print("R^2:", r2_score(y_test, y_pred_poly))


# Splines for Year (train/test included)
spline_train = dmatrix("bs(Year, df=6, degree=3, include_intercept=False)",
                       {"Year": X_train["Year"]}, return_type="dataframe")

spline_test = dmatrix("bs(Year, df=6, degree=3, include_intercept=False)",
                      {"Year": X_test["Year"]}, return_type="dataframe")

spline_model = sm.OLS(y_train, spline_train).fit()
y_pred_spline = spline_model.predict(spline_test)

print("\nSpline Regression (Year)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_spline)))
print("R^2:", r2_score(y_test, y_pred_spline))
print(spline_model.summary())


# Regularization: Ridge & Lasso with CV
ridge = Pipeline([
    ("pre", pre_basic),
    ("reg", Ridge(alpha=1.0))
])

lasso = Pipeline([
    ("pre", pre_basic),
    ("reg", Lasso(alpha=0.001, max_iter=5000))
])

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

print("\nRidge Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("R^2:", r2_score(y_test, y_pred_ridge))

print("\nLasso Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print("R^2:", r2_score(y_test, y_pred_lasso))


# Model Comparison Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lin, alpha=0.6, label="Linear")
plt.scatter(y_test, y_pred_poly, alpha=0.6, label="Polynomial (deg2)")
plt.scatter(y_test, y_pred_ridge, alpha=0.6, label="Ridge")
plt.scatter(y_test, y_pred_lasso, alpha=0.6, label="Lasso")
plt.scatter(y_test, y_pred_spline, alpha=0.6, label="Spline (Year)")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "k--", lw=2)

plt.xlabel("Actual Time (seconds)")
plt.ylabel("Predicted Time (seconds)")
plt.title("Model Comparison: Olympic Swimming Performance")
plt.legend()
plt.axis("equal")
plt.show()


# Residual Plots for Diagnostics
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred_lin, y=y_test - y_pred_lin, lowess=True,
              scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title("Residual Plot: Linear Regression")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()


# QQ-Plot of Residuals
sm.qqplot(y_test - y_pred_lin, line="45")
plt.title("QQ Plot of Linear Regression Residuals")
plt.show()




