# Linear Regression — Home Price Prediction

This folder contains a simple **Linear Regression** example that predicts house price from area (sq ft) using scikit-learn.

---

## Theory

### What is Linear Regression?

**Linear regression** models the relationship between a **dependent variable** (e.g. price) and one or more **independent variables** (e.g. area) as a straight line. The goal is to find the best-fit line that minimizes the error between predicted and actual values.

**Equation (one variable):**

\[
y = m . x + c
\]

- **y** = dependent variable (e.g. price)  
- **x** = independent variable (e.g. area)  
- **m** = slope (coefficient) — how much y changes per unit change in x  
- **c** = intercept — value of y when x = 0  

The model learns **m** and **c** from the training data so that the line fits the points as closely as possible (usually by minimizing **Mean Squared Error**).

---

## Code Snippets and What They Do

### 1. Imports

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
```

- **pandas**: Load and work with the CSV (DataFrame).
- **numpy**: Numerical operations (used internally by sklearn).
- **matplotlib**: Plot scatter and regression line.
- **sklearn.linear_model**: Provides `LinearRegression` for fitting and prediction.

---

### 2. Load the Data

```python
df = pd.read_csv("homeprices.csv")
df
```

- Reads `homeprices.csv` into a DataFrame.
- Columns: `area` (sq ft) and `price`.
- Each row is one house (area, price). This is the **training data** the model will learn from.

---

### 3. Visualize the Data

```python
%matplotlib inline
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(df.area, df.price)
```

- **scatter**: Plots each (area, price) as a point.
- Helps you see if a straight-line relationship is reasonable before fitting the model.

---

### 4. Create and Train the Model

```python
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
```

- **`LinearRegression()`**: Creates an empty linear regression model (no slope/intercept yet).
- **`reg.fit(X, y)`**: Trains the model.
  - **X** = features (here, `df[['area']]` — note the double brackets).
  - **y** = target (here, `df.price`).
- After `fit`, the model has learned **slope** (`reg.coef_`) and **intercept** (`reg.intercept_`).

---

### 5. Predict Price for a New Area

```python
reg.predict([[3300]])
```

- **`predict(X)`**: Uses the fitted line to predict **y** for the given **X**.
- Here we pass **one sample** with **one feature** (area = 3300).
- Returns the predicted price for a 3300 sq ft house (e.g. a single number in an array).

---

### 6. Slope and Intercept

```python
reg.coef_    # slope (m)
reg.intercept_   # intercept (c)
```

- **`reg.coef_`**: Learned slope — “price per unit area”.
- **`reg.intercept_`**: Learned intercept — baseline price when area = 0 (often not meaningful in real terms).
- Together they define the line: **price = coef_ × area + intercept_**.

---

### 7. Plot the Regression Line

```python
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(df.area, df.price)
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
```

- **scatter**: Same training points.
- **plot**: Draws the fitted line by predicting price for each training area — so you see how well the line fits the data.

---

## Why Does `.predict([[3300]])` Take a 2D Array?

In scikit-learn, **all feature inputs are 2D**: shape **(number of samples, number of features)**.

- **One sample** → first dimension = 1.  
- **One feature** (area) → second dimension = 1.  
- So one value must be given as **one row** of a 2D structure: `[[3300]]`.

| Input           | Shape   | Meaning                          |
|----------------|---------|----------------------------------|
| `3300`         | scalar  | Not valid for sklearn            |
| `[3300]`       | (1,)    | 1D — sklearn expects 2D         |
| `[[3300]]`     | (1, 1)  | 1 sample, 1 feature — **correct**|

So:

- **`reg.predict([[3300]])`** → “Predict for **one** house with area **3300**” → returns one predicted price.
- **`reg.predict([[3300], [4000]])`** → “Predict for **two** houses (3300 and 4000 sq ft)” → returns two prices.

This design allows the same API for one or many samples and one or many features, so `.predict()` always receives a 2D array of shape `(n_samples, n_features)`.

---

## Summary

- **Theory**: Linear regression fits a line \(y = mx + c\) to predict a target from one or more features.
- **Code**: Load data → plot → `LinearRegression()` → `fit(areas, prices)` → `predict([[new_area]])` and inspect `coef_` / `intercept_`.
- **2D input**: `.predict([[3300]])` is used because sklearn expects features in 2D form **(samples × features)**; here that is one sample and one feature.
