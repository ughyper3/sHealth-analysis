from math import sqrt, log, exp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

from matplotlib.pyplot import subplots
from pandas import read_csv, DataFrame, crosstab
from process import ProcessDataset
from seaborn import pairplot, lineplot, pointplot, histplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection
from scipy.stats import chi2_contingency



class Shealth:

    process = ProcessDataset()
    raw_data = read_csv('shealth.csv',  sep=";", decimal=',')
    data_set = process.process_data(raw_data)

    B = data_set["day"]
    data_set_without_categorical_variables = data_set.drop(['day', 'date', 'country_number', 'week'], axis=1)

    scaler = StandardScaler()
    data_set_scale = scaler.fit_transform(data_set_without_categorical_variables)
    n_components = 2
    pca = PCA(n_components=n_components)
    data_set_without_categorical_variables_proj = pca.fit_transform(data_set_scale)
    variance_of_first_components = pca.explained_variance_ratio_
    data_set_without_categorical_variables_proj2 = pd.DataFrame(data=data_set_without_categorical_variables_proj,
                                                                columns=['PC1', 'PC2'])
    concat_data_for_PCA = pd.concat([data_set_without_categorical_variables_proj2, B], axis=1)
    pcs = pca.components_




    def display_circles(pcs, n_components, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
        for d1, d2 in axis_ranks:  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
            if d2 < n_components:

                # initialisation de la figure
                fig, ax = plt.subplots(figsize=(7, 6))

                # détermination des limites du graphique
                if lims is not None:
                    xmin, xmax, ymin, ymax = lims
                elif pcs.shape[1] < 30:
                    xmin, xmax, ymin, ymax = -1, 1, -1, 1
                else:
                    xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])

                # affichage des flèches
                # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
                if pcs.shape[1] < 30:
                    plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                               pcs[d1, :], pcs[d2, :],
                               angles='xy', scale_units='xy', scale=1, color="grey")
                    # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
                else:
                    lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                    ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

                # affichage des noms des variables
                if labels is not None:
                    for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                            plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation,
                                     color="blue", alpha=0.5)

                # affichage du cercle
                circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)

                # définition des limites du graphique
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)

                # affichage des lignes horizontales et verticales
                plt.plot([-1, 1], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-1, 1], color='grey', ls='--')

                # nom des axes, avec le pourcentage d'inertie expliqué
                plt.xlabel(
                    'Composant Principal{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
                plt.ylabel(
                    'Composant Principal{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

                plt.title("Cercle des corrélations des (Composants Principaux{} et {})".format(d1 + 1, d2 + 1))
                plt.show(block=False)

    display_circles(pcs, n_components, pca, [(0, 1)], labels=np.array(data_set_without_categorical_variables.columns.tolist()))

    data_set_description = data_set.describe()
    day = data_set['day']
    date = data_set['date']
    weight = data_set['weight']
    steps = data_set['steps']
    walk_duration = data_set['walk_duration']
    sport_duration = data_set['sport_duration']
    spent_energy = data_set['spent_energy']
    country_number = data_set['country_number']
    speed = data_set['average_speed']

    year_aggregation = data_set.groupby(date.dt.to_period('Y')).agg('mean')
    month_aggregation = data_set.groupby(date.dt.to_period('M')).agg('mean')
    week_aggregation = data_set.groupby(date.dt.to_period('w')).agg('mean')
    weekday_aggregation = data_set.groupby(day).agg('mean')

    date_min = date.min()
    date_max = date.max()
    correlation = data_set.corr()

    @staticmethod
    def get_confidence_interval(data_set: DataFrame, correlation: float) -> list:
        """
        This method return the confidence interval between a correlation
        :param data_set: with the dataset we have access to dataset.shape
        :param correlation: coef of correlation (i.e one case of the correlation matrix)
        :return: return a len 2 list with the lower bound and the upper bound of the confidence interval
        """
        if not correlation == 1:
            s = sqrt(1 / (data_set.shape[0] - 3))
        z = (log(1 + correlation) - log(1 - correlation)) / 2
        z_low = z - 1.96 * s
        z_up = 1 + 1.96 * s
        lower_bound = (exp(2 * z_low) - 1) / (exp(2 * z_low) + 1)
        upper_bound = (exp(2 * z_up) - 1) / (exp(2 * z_up) + 1)
        interval = [lower_bound, upper_bound]
        interval.sort()
        return interval

    def display_correlation_confidence_interval(self):
        """
        Display on each line the combination of two attributes
        and the linked confidence interval of there correlation coef
        :return: void
        """
        for i in self.correlation:
            for j in self.correlation:
                if not j == i:
                    interval = f'{i} : {j} : {self.get_confidence_interval(self.data_set, self.correlation[i][j])}'
                    print(interval)

    def display_pair_plot(self):
        """
        :return: Correlation pair plot of the dataset
        """
        return pairplot(self.data_set)

    def display_graph_date_steps(self):
        """
        :return: Graph of the number of steps by date
        """
        return lineplot(x=self.date, y=self.steps)

    def display_graph_year_steps(self):
        """
        :return: Graph of the number of avg steps by year
        """
        return self.year_aggregation['steps'].plot()

    def display_graph_month_steps(self):
        """
        :return: Graph of the number of avg steps by month
        """
        return self.month_aggregation['steps'].plot()

    def display_graph_week_steps(self):
        """
        :return: Graph of the number of avg steps by week
        """
        return self.week_aggregation['steps'].plot()

    def display_graph_weekday_steps(self):
        """
        :return: Graph of the number of avg steps by week day
        """
        return self.weekday_aggregation['steps'].plot()

    def display_graph_date_speed(self):
        """
        :return: Graph of the number of steps by date
        """
        return lineplot(x=self.date, y=self.speed)

    def display_graph_year_speed(self):
        """
        :return: Graph of the number of avg steps by year
        """
        return self.year_aggregation['average_speed'].plot()

    def display_graph_month_speed(self):
        """
        :return: Graph of the number of avg steps by month
        """
        return self.month_aggregation['average_speed'].plot()

    def display_graph_week_speed(self):
        """
        :return: Graph of the number of avg steps by week
        """
        return self.week_aggregation['average_speed'].plot()

    def display_graph_weekday_speed(self):
        """
        :return: Graph of the number of avg steps by week day
        """
        return self.weekday_aggregation['average_speed'].plot()

    def display_variance_explained_by_first_components(self):
        """
        :return: Graph of the explained variance by first pca components
        """
        (fig, ax) = subplots(figsize=(8, 6))
        pointplot(x=[i for i in range(1, self.n_components + 1)], y=self.variance_of_first_components)
        ax.set_title('Variance explained by components')
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Explained Variance')

    def display_hist_weight_and_day(self):
        """
        :return: Graph of the histograms about weight and days
        """
        plt_1 = plt.figure(figsize=(10, 5))
        histplot(self.data_set, x="weight", multiple="dodge", hue="day")
        plt_1.show()
        plt_2 = plt.figure(figsize=(10, 5))
        histplot(self.data_set, x="day", multiple="dodge", hue="weight")
        plt_2.show()

    def display_crosstab_weight_and_day(self):
        """
        :return: cross tab about weight and days
        """
        crosstab_weight_day = crosstab(self.data_set['weight'], self.data_set['day'])
        return crosstab_weight_day

    def display_chi2_cramer_crosstab_cramer_weight_and_day_and_deglib(self):
        """
        :return: cross tab about weight and days and his "degré de liberté" and CRAMER
        """
        crosstab_weight_day = crosstab(self.data_set['weight'], self.data_set['day'])
        deg_lib_crosstab_weight_and_day = ((crosstab_weight_day.shape[0] - 1) * (crosstab_weight_day.shape[1] - 1))
        print(deg_lib_crosstab_weight_and_day)
        chi2_weight_and_day = chi2_contingency(crosstab_weight_day)[0]
        sample_size = sum(crosstab_weight_day.sum())
        minimum_dimension_weight_and_day = min(crosstab_weight_day.shape) - 1


        cramer_day_and_weight = sqrt(chi2_weight_and_day / (sample_size*(minimum_dimension_weight_and_day)))
        return cramer_day_and_weight

