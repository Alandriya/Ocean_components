import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
    :param files_path_prefix: path to the working directory
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
    :param files_path_prefix: path to the working directory
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
    :param files_path_prefix: path to the working directory
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
    :param files_path_prefix: path to the working directory
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


def plot_A_B_coefficients(files_path_prefix, a_list, b_list):
    min_a = [min(a_list[i]) if a_list[i] else None for i in range(len(a_list))]
    mean_a = [np.nanmean(a_list[i]) for i in range(len(a_list))]
    a_90 = [np.quantile(a_list[i], 0.90) if a_list[i] else None for i in range(len(a_list))]
    a_95 = [np.quantile(a_list[i], 0.95) if a_list[i] else None for i in range(len(a_list))]
    max_a = [np.nanmax(a_list[i]) for i in range(len(a_list))]

    min_b = [np.min(list(filter(lambda x: x != None, b_list[i]))) for i in range(len(b_list))]
    mean_b = [np.mean(list(filter(lambda x: x != None, b_list[i]))) for i in range(len(b_list))]
    b_90 = [np.quantile(list(filter(lambda x: x != None, b_list[i])), 0.90) for i in range(len(b_list))]
    b_95 = [np.quantile(list(filter(lambda x: x != None, b_list[i])), 0.95) for i in range(len(b_list))]
    max_b = [np.max(list(filter(lambda x: x != None, b_list[i]))) for i in range(len(b_list))]

    plt.figure(figsize=(10, 10))
    plt.plot(min_a, '-', label='min a', c='plum')
    plt.plot(mean_a, label='mean a', c='r')
    plt.plot(a_90, '--', label='a 90', c='violet')
    plt.plot(a_95, '--', label='a 95', c='fuchsia')
    plt.plot(max_a, '-', label='max a', c='pink')
    plt.legend()
    plt.savefig(files_path_prefix + f'A.png')

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.plot(min_b, '-', label='min b', c='bisque')
    plt.plot(mean_b, label='mean b', c='y')
    plt.plot(b_90, '--', label='b 90', c='olive')
    plt.plot(b_95, '--', label='b 95', c='orange')
    plt.plot(max_b, '-', label='max b', c='khaki')
    plt.legend()
    plt.savefig(files_path_prefix + f'B.png')
    return