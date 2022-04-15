import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate


def generating_data_and_making_interpolation(f1, f2, n_of_left_samples, n_of_right_samples, interpolation_kind,
                                             using_cubic=False):
    left_part_samples = np.linspace(-2.5, 0, n_of_left_samples)
    right_part_samples = np.linspace(0.001, 1.5, n_of_right_samples)

    left_part_values = f1(left_part_samples)
    right_part_values = f2(right_part_samples)

    x_samples = list(left_part_samples) + list(right_part_samples)
    y_samples = list(left_part_values) + list(right_part_values)

    M = (n_of_right_samples + n_of_left_samples) * 5

    if not using_cubic:

        interpolation_object = interpolate.interp1d(x_samples, y_samples, kind=interpolation_kind)

        x_resampled_l = np.linspace(-2.5, 0, int(2 * M / 3))
        x_resampled_r = np.linspace(0, 1.5, int(M / 3))

        x_resampled = list(x_resampled_l) + list(x_resampled_r)
        y_resampled = interpolation_object(x_resampled)

        y_from_function_l = f1(x_resampled_l)
        y_from_function_r = f2(x_resampled_r)

        y_from_function = list(y_from_function_l) + list(y_from_function_r)

    else:
        interpolation_object = interpolate.CubicSpline(x_samples, y_samples)

        x_resampled_l = np.linspace(-2.5, 0, int(2 * M / 3))
        x_resampled_r = np.linspace(0, 1.5, int(M / 3))

        x_resampled = list(x_resampled_l) + list(x_resampled_r)
        y_resampled = interpolation_object(x_resampled)

        y_from_function_l = f1(x_resampled_l)
        y_from_function_r = f2(x_resampled_r)

        y_from_function = list(y_from_function_l) + list(y_from_function_r)

    return x_samples, y_samples, x_resampled, y_from_function, y_resampled


def drawing_plot(x_samples, y_samples, x_resampled, y_from_function, y_resampled, settings):
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(x_resampled, y_from_function, color='#DE3163', label="funkcja wejściowa", linewidth=4)
    ax.plot(x_samples, y_samples, 'h', color='#6495ED', label='próbki', markersize=15, markeredgecolor='black')
    ax.plot(x_resampled, y_resampled, '--', color='#9C27B0', label=settings['label'], markersize=5, linewidth=3)

    ax.set_xticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_title(settings['title'])

    ax.legend(shadow=True)

    plt.show()


spline_settings = {'label': 'Wykres interpolacji funkcjami sklejanymi 3 stopnia',
                   'title': 'Interpolacja funkcjami sklejanymi 3 stopnia'}

linear_settings = {'label': 'Wykres interpolacji liniowej',
                   'title': 'Interpolacja liniowa'}

third_degree_settings = {'label': 'Wykres interpolacji wielomianowej 3 stopnia',
                         'title': 'Interpolacja wielomianowa 3 stopnia'}

settings = [linear_settings, spline_settings, third_degree_settings]

lpe = lambda x: (-1 / (x - 1))
rpe = lambda x: -0.4 * x ** 2 + 1
langrange = lambda x: x ** 3 + 5 * x ** 2 - 2

for index, kind in enumerate(['linear', 'cubic', 'cubic_f']):

    if kind != 'cubic_f':
        xs, ys, xr, yff, yr = generating_data_and_making_interpolation(lpe, rpe, 8, 4, kind, False)
        drawing_plot(xs, ys, xr, yff, yr, settings[index])
    else:
        xs, ys, xr, yff, yr = generating_data_and_making_interpolation(lpe, rpe, 8, 4, kind, True)
        drawing_plot(xs, ys, xr, yff, yr, settings[index])
