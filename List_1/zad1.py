# Lista 1 zad 1 regresja liniowa
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


x, y = datasets.make_regression(n_samples=200, n_features=1, noise=12)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

x_plot = list(np.linspace(start=x.min(), stop=x.max(), num=300))
print(x_plot)
print(y_pred)

plt.scatter(x_train, y_train, label='dane treningowe', color='black')
plt.scatter(x_test, y_test, edgecolor='black', facecolor='white', label='dane testujące')
plt.plot(x_test, y_pred, color="red", linewidth=3, label='model liniowy')


x_ticks = np.linspace(min(x), max(x), 11)
y_ticks = np.linspace(min(y), max(y), 11)

plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.xlabel('Wartości "x"', fontsize=14)
plt.ylabel('Wartości "y"', fontsize=14)

plt.legend()
plt.show()