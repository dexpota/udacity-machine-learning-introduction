[linear-regression]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
[polynomial-features]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
[lasso]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
[standard-scaler]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

# Linear Regression

## TL;DR

TODO: to be filled with a summary

## Notes

TODO: to be filled with notes

## Snippets

> A collection of the most important snippets about linear regression in
> `scikit.learn`

### Linear Regression Model

Creating a linear regression model in `scikit.learn` is easy as 123 with the
`LinearRegression` class. Follow the the steps in the snippet below.

```python
model = LinearRegression()

# 1. Fit your model with your training data
model.fit(X, y)
# where X is an array of n_samples by n_features and y is an array of n_samples
# by n_targes (your output)

# 2. Predict using the linear model
model.predict(x)
# where x is an array of n_samples by n_features
```

### Polynomial Features

With the linear regression method we can create models consisting of linear
combinations of polynomial functions with the help of `PolynomialFeatures`.

```python
# 1. Create an instance of PolynomialFeatures and specify the degree
poly_features = PolynomialFeatures(4)
# 2. Create the polynomial features
X_poly = poly_features.fit_transform(X)
# 3. User LinearRegression to create a linear model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
```

### Lasso

It is possible to regularize the coefficients and control our model complexity
with some help from the `Lasso` class.

```python
# 1. Create an instance of Lasso
lasso_reg = Lasso()
# 2. Fit your model with your training data
lasso_reg.fit(X, y)
# 3. Make some prediction
lasso_reg.predict(x)
# Optional. Take a look at the coefficient values
lasso_reg.coef_
```

### Feature Scaling

Any time you choose to use `Lasso` or any other regularization methods you will
want to scale the features with `StandardScaler`.

```python
# 1. Scale your feature vector X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 2. Use Lasso to fit your model and make prediction
lasso_reg = Lasso()
lasso_reg.fit(X_scaled, y)
```

### References

- [LinearRegression][linear-regression];
- [PolynomialFeatures][polynomial-features];
- [Lasso][lasso];
- [StandardScaled][standard-scaler];
