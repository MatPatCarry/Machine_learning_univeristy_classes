import sklearn
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
from sklearn.model_selection import train_test_split

a = -10
b = 50

x_uniform = np.random.sample(200)

x_values = (b-a) * x_uniform + a
y_values = []
y_values_with_z = []
a = 2
b = 5

for x_value in x_values:

    y_value = a * x_value + b
    y_values.append(y_value.__round__(2))

for y in y_values:
    z_uniform = np.random.uniform()
    z = 14 * z_uniform + 1
    plus_or_minus = np.random.randint(2)
    sign_dict = {0: 1, 1: -1}
    # print(z)
    y_values_with_z.append(y + z * sign_dict[plus_or_minus])


x_train, x_test, y_train, y_test = train_test_split(x_values, y_values_with_z, train_size=0.75)

model = linear_model.Ridge(alpha=0.05)
model.fit(x_train.reshape(-1, 1), y_train)
y_pred = model.predict(x_test.reshape(-1, 1))

print(model.coef_[0], model.intercept_)

x_values.sort()
y_values.sort()

x_plot = list(np.linspace(start=x_values.min(), stop=x_values.max(), num=300))


plt.scatter(x_train, y_train, label='dane treningowe', color='black')
plt.scatter(x_test, y_test, edgecolor='black', facecolor='white', label='dane testujące')
plt.plot(x_test, y_pred, color="red", linewidth=3, label=f'model regresji liniowej z regularyzacją l2:\ny = {round(model.coef_[0], 2)}x + {round(model.intercept_, 2)}')
plt.plot(x_values, y_values, color='blue', linewidth=3, label=f'początkowa prosta o równaniu:\ny = {a}x + {b}')


x_ticks = np.linspace(min(x_values), max(x_values), 11)
y_ticks = np.linspace(min(y_values_with_z), max(y_values_with_z), 11)

plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.xlabel('Wartości "x"', fontsize=14)
plt.ylabel('Wartości "y"', fontsize=14)

plt.legend()
plt.show()