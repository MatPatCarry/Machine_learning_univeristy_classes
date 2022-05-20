
def plotting_predictions(predictions, *, train_set, test_set, pred_type='predictions', **settings_dict):
    
    import matplotlib.pyplot as plt
    import copy

    default_settings = {}

    n_train_set_samples_to_display = None
    model_type = 'MA'
    y_lim = None
    x_label = ''
    y_label = 'Values'

    default_settings['n_train_samples'] = n_train_set_samples_to_display
    default_settings['model_type'] = model_type
    default_settings['y_lim'] = y_lim
    default_settings['x_label'] = x_label 
    default_settings['y_label'] = y_label

    if settings_dict:
        
        try:

            for key, value in settings_dict.items():
                default_settings[key] = value

        except KeyError:
            return 'Incorrect key word arguments!'

    if default_settings['n_train_samples']:
        train_set_to_display = copy.deepcopy(train_set[-int(default_settings['n_train_samples']):])
    else:
        train_set_to_display = copy.deepcopy(train_set)

    if pred_type == 'predictions':

        fig, ax1 = plt.subplots(1, 1, figsize=(19, 9))

        ax1.plot(train_set_to_display, color = '#FF4500', label="Original data - train set", linewidth=3)
        ax1.plot(predictions, color = '#9370DB' , label=f"{default_settings['model_type']} model predictions", linewidth=3)

        ax1.set_xlabel(default_settings['x_label'])
        ax1.set_ylabel(default_settings['y_label'], fontsize=15)
        ax1.legend(loc='best', fontsize=15) 

    elif pred_type == 'forecast':

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))

        if y_lim:
            ax2.set_ylim(y_lim)

        ax1.plot(train_set_to_display, color = '#FF4500', label="Original data - train set", linewidth=2)
        ax1.plot(test_set, color = '#FF8C00', label="Original data - test set", linewidth=2)

        ax2.plot(predictions, color = '#9370DB', label=f"{model_type} model forecast", linewidth=2)
        ax2.plot(test_set[1:], color = '#FF8C00', label="Original data - test set", linewidth=2)

        ax3.plot(train_set_to_display, color = '#FF4500', label="Original data - train set", linewidth=2)
        ax3.plot(test_set, color = '#FF8C00', label="Original data - test set", linewidth=2)
        ax3.plot(predictions, color = '#9370DB', label=f"{model_type} model forecast", linewidth=2)

        for ax in (ax1, ax2, ax3):
            ax.legend(loc='best')
            ax.set_xlabel(default_settings['x_label'])
            ax.set_ylabel(default_settings['y_label'])

    fig.suptitle(f"{default_settings['model_type']} model {pred_type}", fontsize=20)

    plt.show()


def comparing_results(real_data, models_results, **settings):

    import matplotlib.pyplot as plt

    default_settings = {}
    y_label = 'Values'

    default_settings['y_label'] = y_label

    try:
        for key, value in settings.items():
            default_settings[key] = value
    except KeyError:
        return 'Incorrect key word arguments!'

    fig, axes = plt.subplots(1, 1, figsize=(25, 10))
    line_colors = ['#9370DB', '#EE82EE', '#4682B4', '#0000CD', '#800000', '#7CFC00', '#FFD700']

    axes.plot(real_data, color='#FF4500', label=f'REAL VALUES', linewidth=4)

    for inx, (model, result) in enumerate(models_results.items()):

        axes.plot(result, color=line_colors[inx], label=f'{str(model).upper()}', linewidth=4, linestyle='--')

    fig.suptitle(f'Comparing models', fontsize=20)
    axes.legend(fontsize=15, loc='best')
    axes.set_ylabel(default_settings['y_label'])

    plt.show()
