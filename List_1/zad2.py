import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

a = -10
b = 50

x_uniform = np.random.sample(200)

x_values = (b-a) * x_uniform + a
y_values = []
y_values_with_z = []
a = 3
b = 5
c = 10

for x_value in x_values:

    y_value = a * (x_value ** 2) + b * x_value + c
    y_values.append(y_value.__round__(2))

for y in y_values:
    z_uniform = np.random.uniform()
    z = (1100) * z_uniform + 1
    plus_or_minus = np.random.randint(3)
    sign_dict = {0: 1, 1: -1, 2: 1}
    # print(z)
    y_values_with_z.append(y + z * sign_dict[plus_or_minus])

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values_with_z, train_size=0.75)

model_GLM = LinearRegression()
gen_features = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
model_GLM.fit(gen_features.fit_transform(x_train.reshape(-1, 1)), y_train)
print(f'Model GLM params: {np.round(model_GLM.coef_,4)}, {np.round(model_GLM.intercept_,5)}')

x_plot = np.linspace(start=x_values.min(), stop=x_values.max(), num=20)
y_GLM_pred_plot = model_GLM.predict(gen_features.fit_transform(x_plot.reshape(-1,1)))


plt.scatter(x_train, y_train, label='dane treningowe', color='black')
plt.scatter(x_test, y_test, edgecolor='black', facecolor='white', label='dane testujące')
plt.plot(x_plot, y_GLM_pred_plot, color="red", linewidth=3, label='model GLM')

x_values.sort()
y_values.sort()

plt.plot(x_values, y_values, color="green", linewidth=3, label='y = 3x² + 5x + 10')

plt.xlabel('Wartości "x"', fontsize=14)
plt.ylabel('Wartości "y"', fontsize=14)
plt.legend()

plt.show()