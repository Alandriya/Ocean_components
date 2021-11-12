import matplotlib
import matplotlib.pyplot as plt

font = {'size': 10}
matplotlib.rc('font', **font)
figsize = (15, 5)


def plot_components(files_path_prefix,
                    date,
                    type_string,
                    series_name,
                    history,
                    indexes,
                    comp_amount,
                    postfix=None,
                    colors=None):
    """
    Plots components on the same plot, one plot for means and one for variances
    :param date: date of program running for creating folder name where to save plots
    :param type_string: type of data for creating folder name where to save plots, string
    :param series_name: name of a series for creating folder name where to save plots, string
    :param history: DataFrame with history of components' evolution in time
    :param indexes: list with indexes of components for matching components with previous plots
    :param comp_amount: int, the amount of extracted components by EM
    :param postfix: postfix added for pictures names, string or None
    :param colors: list with colors for each component or None, for matching components with previous plots
    :return:
    """

    def plot_param(param_name):
        plt.figure(figsize=figsize)
        for k in indexes:
            time_points = list()
            values = list()
            for t in history.index:
                for i in range(comp_amount):
                    if history.loc[t, f'label_{i + 1}'] == k:
                        time_points.append(t)
                        values.append(history.loc[t, f'{param_name}_{i + 1}'])
            if colors is not None:
                plt.plot(time_points, values, "-o", label=f'{param_name} {k}', col=colors[k])
            else:
                plt.plot(time_points, values, "-o", label=f'{param_name} {k}')
        plt.legend()
        if postfix:
            plt.savefig(files_path_prefix + f'prepared/new_components/{series_name}_{param_name}s.png')
        else:
            # plt.savefig(files_path_prefix + f'prepared/new_components/{series_name}_{param_name}s.png')
            plt.show()
        plt.clf()

    plot_param('mean')
    plot_param('sigma')
    plot_param('weight')
    return


def plot_clusters(files_path_prefix,
                  date,
                  type_string,
                  series_name,
                  data,
                  labels,
                  postfix=''):
    """
    Plots clusters on (a, sigma) space, where a is mean and sigma is variance. Points are created for every component
    for every time moment
    :param date: date of program running for creating folder name where to save plots
    :param type_string: type of data for creating folder name where to save plots, string
    :param series_name: name of a series for creating folder name where to save plots, string
    :param data:
    :param labels:
    :param postfix: postfix added for picture names, string
    :return:
    """
    # font = {'weight': 'bold', 'size': 26}
    matplotlib.rc('font', **font)
    figsize = (7, 7)
    plt.figure(figsize=figsize)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title('Mean - sigma')
    # plt.savefig(files_path_prefix + f'prepared/new_components/{series_name}_clusters.png')
    plt.show()
    plt.clf()

    if data.shape[1] > 2:
        plt.figure(figsize=figsize)
        plt.scatter(data[:, 0], data[:, 2], c=labels)
        plt.title('Mean - weight')
        plt.show()
        plt.clf()

        plt.figure(figsize=figsize)
        plt.scatter(data[:, 1], data[:, 2], c=labels)
        plt.title('Sigma - weight')
        plt.show()
        plt.clf()

    return


def plot_integrated_components(files_path_prefix,
                               date,
                               type_string,
                               series_name,
                               components_df,
                               series_df,
                               comp_numbers,
                               width=10000,
                               postfix_param='',
                               plot_each_component=False):
    """
    Plots integrated components (optionally) and final total component comparing with moving average and real time
    series values on the same plot.
    :param date: date of program running for creating folder name where to save plots
    :param type_string: type of data for creating folder name where to save plots, string
    :param series_name: name of a series for creating folder name where to save plots, string
    :param components_df: DataFrame with integrated components' history
    :param series_df: DataFrame with time series and moving average
    :param comp_numbers: list with indexes of components for matching components with previous plots
    :param width: width of the window
    :param postfix_param:
    :param plot_each_component: if plot additionally every component in a several plot
    :return:
    """
    plt.figure(figsize=figsize)
    for part in range(len(components_df) // width):
        part_components = components_df[part * width:(part + 1) * width]
        part_series = series_df[part * width:(part + 1) * width]
        if postfix_param:
            postfix = f'({part * width}-{(part + 1) * width})' + postfix_param
        else:
            postfix = f'({part * width}-{(part + 1) * width})'

        if plot_each_component:
            for comp in comp_numbers:
                plt.figure(figsize=figsize)
                plt.plot(part_series['time'], part_components[f'integrated_{comp}'], label=f'integrated_{comp}')
                plt.legend()
                plt.savefig(files_path_prefix + f'prepared/new_components/{series_name}_integrated.png')
                plt.clf()

        plt.plot(part_series['time'], part_series['value'], c='b', label='Source')
        plt.plot(part_series['time'], part_series['moving_average'], c='g', label='Moving average')
        plt.plot(part_series['time'], part_components['final'], c='r', label='Expectation summ')
        plt.legend()
        plt.savefig(files_path_prefix + f'prepared/new_components/{series_name}_ma_versus_summ{postfix}.png')
        plt.clf()

    return


def plot_rmse(files_path_prefix,
              date,
              type_string,
              series_name,
              rmse_df,
              postfix=''):
    """
    Plot total errors for integrated final component and moving average, comparing to real values of time series
    :param date: date of program running for creating folder name where to save plots
    :param type_string: type of data for creating folder name where to save plots, string
    :param series_name: name of a series for creating folder name where to save plots, string
    :param rmse_df: DataFrame with pre-counted errors
    :param postfix:
    :return:
    """
    # font = {'weight': 'bold', 'size': 14}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(15, 15))
    x = range(len(rmse_df))
    plt.xticks(x, rmse_df['period'])

    plt.plot(x, rmse_df['moving average'], c='g', label='Moving average')
    plt.plot(x, rmse_df['integrated'], c='r', label='Integrated')
    plt.legend()
    plt.savefig(files_path_prefix + f'prepared/new_components/{series_name}_RMSE{postfix}.png')
    plt.clf()
