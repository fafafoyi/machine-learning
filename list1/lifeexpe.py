import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


""" leds = life expectancy data set """
leds = pd.read_csv("LifeExpectancy.csv")

leds_clean = leds.dropna()

train_side, test_side = train_test_split (leds_clean , test_size=0.2, random_state=42)

sns.set(style="whitegrid")

latest_year_data = leds_clean.sort_values(by="Year", ascending=False).drop_duplicates(subset="Country")

top_countries = latest_year_data[["Country", "Life expectancy "]].sort_values(by="Life expectancy ", ascending=False).head(3)



def simple_linear_regression(feature_name, target_name='Life expectancy '):
    data = train_side[[feature_name, target_name]].dropna()

    X = data[[feature_name]].values
    y = data[target_name].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    score = model.score(X, y)

    print(f"\n=== Linear Regression: {feature_name} ===")
    print(f"Equation: Life Expectancy = {slope:.4f} * {feature_name} + {intercept:.4f}")
    print(f"R² Score: {score:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='lightcoral', label='Data Points')
    plt.plot(X, model.predict(X), color='blue', linewidth=2, label='Regression Line')
    plt.title(f'Life Expectancy vs {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Life Expectancy')
    plt.legend()
    eq_text = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))
    plt.grid(True)
    plt.show()

    return model

def evaluate_model(model, feature_name, test_df, target_name='Life expectancy '):
    test_data = test_df[[feature_name, target_name]].dropna()

    X_test = test_data[[feature_name]].values
    y_true = test_data[target_name].values
    y_pred = model.predict(X_test)

    errors = y_pred - y_true
    mean_error = errors.mean()
    std_error = errors.std()

    print(f"\n=== Evaluation: {feature_name} ===")
    print(f"Average Error: {mean_error:.4f}")
    print(f"Standard Deviation of Error: {std_error:.4f}")

    return errors


life_stats = train_side['Life expectancy '].describe()
print("Statistical Summary of Life Expectancy:")
print(life_stats)
print("Standard Deviation:", train_side['Life expectancy '].std())
print("Top 3 countries with the highest life expectancy:")
print(top_countries)


plt.figure(figsize=(10,6))
sns.histplot(train_side['Life expectancy '], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Life Expectancy (Training Set)")
plt.xlabel("Life Expectancy")
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


model_gdp = simple_linear_regression('GDP')
model_expenditure = simple_linear_regression('Total expenditure')
model_alcohol = simple_linear_regression('Alcohol')

errors_gdp = evaluate_model(model_gdp, 'GDP', test_side)
errors_expenditure = evaluate_model(model_expenditure, 'Total expenditure', test_side)
errors_alcohol = evaluate_model(model_alcohol, 'Alcohol', test_side)

"""Part2"""
features = ['Schooling', 'Income composition of resources', 'Polio', 'Adult Mortality']
target = 'Life expectancy '

train_multi = train_side[features + [target]].dropna()
test_multi = test_side[features + [target]].dropna()

X_train = train_multi[features]
y_train = train_multi[target]
X_test = test_multi[features]
y_test = test_multi[target]

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

print("\n=== Multivariate Regression Coefficients ===")
for name, coef in zip(features, multi_model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {multi_model.intercept_:.4f}")

score = multi_model.score(X_train, y_train)
print(f"Training R² Score: {score:.4f}")

# Make predictions
y_pred = multi_model.predict(X_test)

# Error metrics
errors = y_pred - y_test
mean_error = errors.mean()
std_error = errors.std()

print("\n=== Prediction Evaluation on Test Set ===")
print(f"Average Error: {mean_error:.4f}")
print(f"Standard Deviation of Error: {std_error:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='black')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs Predicted Life Expectancy (Multivariate)')
plt.grid(True)
plt.show()

