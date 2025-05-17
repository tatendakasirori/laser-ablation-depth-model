import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 5/16/2025 data x= depth, y= mass
# Need more data as model is underfitting 
x = np.array([7, 11, 16, 19, 23, 27, 31, 34, 38, 43]).reshape(-1, 1)
y = np.array([ 27.4694, 31.1945, 34.8253, 36.3726, 39.6916, 43.0477, 46.2648, 48.8145, 52.2542, 55.7755 ])

# 2. Split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=1
)

# 3. Build a pipeline: polynomial features + linear regression
degree = 3
model = make_pipeline(
    PolynomialFeatures(degree, include_bias=False),
    LinearRegression()
)

# 4. Train the model
model.fit(x_train, y_train)

# 5. Make predictions
y_pred_train = model.predict(x_train)
y_pred_test  = model.predict(x_test)

# 6. Evaluate
r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test,  y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)

# 7. Output results
print(f"Polynomial degree: {degree}")
print(f"Training   R²: {r2_train:.4f}")
print(f"Test       R²: {r2_test:.4f}")
print(f"Test   MSE: {mse_test:.4f}")

# 8. Inspect learned coefficients
#    model.named_steps['linearregression'] is the fitted regressor
coef = model.named_steps['linearregression'].coef_
intercept = model.named_steps['linearregression'].intercept_
print(f"Learned model: y = {intercept:.4f} + ({coef[0]:.4f})x + ({coef[1]:.4f})x^2 + ({coef[2]:.4f})x^3")
