import sklearn
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from gurobipy import *
import scipy

a = -10
b = 50

x_uniform = np.random.sample(200)

x_values = (b-a) * x_uniform + a
y_values = []
y_values_with_z = []
a = 2
b = 5
c = 4

for x_value in x_values:

    y_value = a * x_value**2 + b * x_value + c
    y_values.append(y_value.__round__(2))

for y in y_values:
    z_uniform = np.random.uniform()
    z = 300 * z_uniform + 1
    plus_or_minus = np.random.randint(2)
    sign_dict = {0: 1, 1: -1}
    # print(z)
    y_values_with_z.append(y + z * sign_dict[plus_or_minus])


x_train, x_test, y_train, y_test = train_test_split(x_values, y_values_with_z, train_size=0.75)

regression_model = Model('regresja wielomianowa')

a = regression_model.addVar(vtype=GRB.CONTINUOUS, name='a')
b = regression_model.addVar(vtype=GRB.CONTINUOUS, name='b')
c = regression_model.addVar(vtype=GRB.CONTINUOUS, name='c')

less_squares_function = (1 / (2 * len(x_train))) *\
                        (sum([(y_train[index] - (a * (x_train[index])**2 + b * x_train[index]  + c)) ** 2 for index in range(len(x_train))]))
# coef if coef > 0 else -1 *

regression_model.setObjective(less_squares_function, GRB.MINIMIZE)
regression_model.optimize()

new_coeficiants = []

for v in regression_model.getVars():
    new_coeficiants.append(float(v.x).__round__(2))
    print(f'{v.varName} : {v.x}')

y_pred_values = []

for x in x_test:
    y_pred_value = new_coeficiants[0] * x**2 + new_coeficiants[1] * x + new_coeficiants[2]
    y_pred_values.append(y_pred_value)


x_values.sort()
y_values.sort()

x_plot = list(np.linspace(start=x_values.min(), stop=x_values.max(), num=300))

y_pred_values.sort()
x_test.sort()

plt.scatter(x_train, y_train, label='dane treningowe', color='black')
plt.scatter(x_test, y_test, edgecolor='black', facecolor='white', label='dane testujące')
plt.plot(x_test, y_pred_values, color="red", linewidth=3, label=f'model regresji wielomianowej:\ny = {new_coeficiants[0]}x^2 + {new_coeficiants[1]}x + {new_coeficiants[2]}')
plt.plot(x_values, y_values, color='blue', linewidth=3, label=f'początkowe równanie:\ny = {2}x^2 + {5}x + 4')


x_ticks = np.linspace(min(x_values), max(x_values), 11)
y_ticks = np.linspace(min(y_values_with_z), max(y_values_with_z), 11)

plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.xlabel('Wartości "x"', fontsize=14)
plt.ylabel('Wartości "y"', fontsize=14)

plt.legend()
plt.show()