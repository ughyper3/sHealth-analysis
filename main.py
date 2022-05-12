from math import sqrt, log, exp
from matplotlib.patches import Circle
from matplotlib.pyplot import subplots, plot, legend, ylabel, xlabel, title, quiver, text, gca, xlim, ylim, show, \
    figure, scatter
from pandas import read_csv, DataFrame, to_datetime, crosstab, concat
from process import ProcessDataset
from seaborn import pairplot, pointplot, histplot, heatmap, set; set()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix, silhouette_score
from matplotlib.collections import LineCollection
from scipy.stats import chi2_contingency
from numpy import zeros


class Shealth:

    process = ProcessDataset()
    raw_data = read_csv('shealth.csv',  sep=";", decimal=',')
    data_set = process.process_data(raw_data)

    data_set_without_categorical_variables = data_set.drop(['day', 'date', 'country_name', 'week'], axis=1)

    scaler = StandardScaler()
    data_set_scale = scaler.fit_transform(data_set_without_categorical_variables)
    data_set_description = data_set.describe()

    day = data_set['day']
    date = data_set['date']
    weight = data_set['weight']
    steps = data_set['steps']
    walk_duration = data_set['walk_duration']
    sport_duration = data_set['sport_duration']
    spent_energy = data_set['spent_energy']
    country_name = data_set['country_name']
    speed = data_set['average_speed']

    group_by_country_name_count = data_set.groupby('country_name')\
        .agg(steps_mean=('steps', 'mean'),
             measure_count=('country_name', 'size'),
             average_speed=('average_speed', 'mean'),
             data_min=('date', 'min'),
             date_max=('date', 'max'),
             calories=('spent_energy', 'mean'),
             sport_duration=('sport_duration', 'mean'),
             min_weight=('weight', 'min'),
             max_weight=('weight', 'max')
             )\
        .sort_values('measure_count', ascending=False)

    year_aggregation = data_set.groupby(date.dt.to_period('Y')).agg('mean')
    month_aggregation = data_set.groupby(date.dt.to_period('M')).agg('mean')
    week_aggregation = data_set.groupby(date.dt.to_period('w')).agg('mean')
    weekday_aggregation = data_set.groupby(day).agg('mean')

    date_min = date.min()
    date_max = date.max()
    correlation = data_set.corr()

    train_data_set = data_set[date <= to_datetime("2016-08-01", format='%Y-%m-%d')]
    test_data_set = data_set[date > to_datetime("2016-08-01", format='%Y-%m-%d')]

    month_train_data_set = month_aggregation[:17]
    month_test_data_set = month_aggregation[-4:]

    week_train_data_set = week_aggregation[:70]
    week_test_data_set = week_aggregation[-17:]

    n_components = 2
    pca = PCA(n_components=n_components)
    data_set_without_categorical_variables_proj = pca.fit_transform(data_set_scale)
    variance_of_first_components = pca.explained_variance_ratio_
    data_set_without_categorical_variables_proj2 = DataFrame(data=data_set_without_categorical_variables_proj,
                                                                columns=['PC1', 'PC2'])
    concat_data_for_PCA = concat([data_set_without_categorical_variables_proj2, day], axis=1)
    pcs = pca.components_

    def display_circles(self, n_components, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
        for d1, d2 in axis_ranks:  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
            if d2 < n_components:

                # initialisation de la figure
                fig, ax = subplots(figsize=(7, 6))

                # détermination des limites du graphique
                if lims is not None:
                    xmin, xmax, ymin, ymax = lims
                elif self.shape[1] < 30:
                    xmin, xmax, ymin, ymax = -1, 1, -1, 1
                else:
                    xmin, xmax, ymin, ymax = min(self[d1, :]), max(self[d1, :]), min(self[d2, :]), max(self[d2, :])

                # affichage des flèches
                # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
                if self.shape[1] < 30:
                    quiver(zeros(self.shape[1]), zeros(self.shape[1]),
                           self[d1, :], self[d2, :],
                           angles='xy', scale_units='xy', scale=1, color="grey")
                    # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
                else:
                    lines = [[[0, 0], [x, y]] for x, y in self[[d1, d2]].T]
                    ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

                # affichage des noms des variables
                if labels is not None:
                    for i, (x, y) in enumerate(self[[d1, d2]].T):
                        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                            text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation,
                                     color="blue", alpha=0.5)

                # affichage du cercle
                circle = Circle((0, 0), 1, facecolor='none', edgecolor='b')
                gca().add_artist(circle)

                # définition des limites du graphique
                xlim(xmin, xmax)
                ylim(ymin, ymax)

                # affichage des lignes horizontales et verticales
                plot([-1, 1], [0, 0], color='grey', ls='--')
                plot([0, 0], [-1, 1], color='grey', ls='--')

                # nom des axes, avec le pourcentage d'inertie expliqué
                xlabel(
                    'Composant Principal{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
                ylabel(
                    'Composant Principal{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

                title("Cercle des corrélations des (Composants Principaux{} et {})".format(d1 + 1, d2 + 1))
                show(block=False)

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
        plt_1 = figure(figsize=(10, 5))
        histplot(self.data_set, x="weight", multiple="dodge", hue="day")
        plt_1.show()
        plt_2 = figure(figsize=(10, 5))
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

    def display_mds_(self):
        """
        :return: mds about day
        """

        B = self.data_set["day"]

        data_set_without_categorical_variables = self.data_set.drop(['day', 'date', 'country_number', 'week'], axis=1)
        scaler = StandardScaler()
        data_set_scale = scaler.fit_transform(data_set_without_categorical_variables)
        mds = MDS(random_state=0, n_components=2)
        data_set_without_categorical_variables_proj_mds = mds.fit_transform(data_set_scale)

        data_set_without_categorical_variables_proj_mds1 = DataFrame(data=data_set_without_categorical_variables_proj_mds,
                                                                    columns=['PC1', 'PC2'])
        concat_msa_day = concat([data_set_without_categorical_variables_proj_mds1, B], axis=1)
        print(concat_msa_day)

        fig = figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('MDS')
        targets = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        colors = ['red', 'blue', 'green', 'purple', 'grey', 'orange', 'yellow']
        for target, color in zip(targets, colors):
            indice = concat_msa_day["day"] == target
            ax.scatter(concat_msa_day.loc[indice, 'PC1'],
                       concat_msa_day.loc[indice, 'PC2'],
                       c=color, s=50)
        ax.legend(targets)
        ax.grid()
        show()

    def display_heat_map(self):
        """
        :return: heatmap visualization
        """
        return heatmap(self.data_set_without_categorical_variables.corr())

    def display_kmeans_(self):
        """
        :return: kmeans...
        """

        B = self.data_set["day"]

        data_set_without_categorical_variables = self.data_set.drop(['day', 'date', 'country_number', 'week'], axis=1)
        scaler = StandardScaler()
        data_set_scale = scaler.fit_transform(data_set_without_categorical_variables)

        for i in list(range(0, 10)):
            kmeans = KMeans(n_clusters=7, n_init=7, max_iter=300, random_state=i).fit(data_set_scale)
            kmeans.score(data_set_scale)
            prediction = kmeans.predict(data_set_scale)

            fig = figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            scatter(data_set_scale[prediction == 0, 0], data_set_scale[prediction == 0, 1], s=50, c='red', label='Monday')
            scatter(data_set_scale[prediction == 1, 0], data_set_scale[prediction == 1, 1], s=50, c='blue', label='Tuesday')
            scatter(data_set_scale[prediction == 2, 0], data_set_scale[prediction == 2, 1], s=50, c='green', label='Wednesday')
            scatter(data_set_scale[prediction == 3, 0], data_set_scale[prediction == 3, 1], s=50, c='purple',
                        label='Thursday')
            scatter(data_set_scale[prediction == 4, 0], data_set_scale[prediction == 4, 1], s=50, c='gray',
                        label='Friday')
            scatter(data_set_scale[prediction == 5, 0], data_set_scale[prediction == 5, 1], s=50, c='orange',
                        label='Saturday')
            scatter(data_set_scale[prediction == 6, 0], data_set_scale[prediction == 6, 1], s=50, c='yellow',
                        label='Sunday')

            scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow',
                        label='Centroids')
            legend()

        kmeans1 = KMeans(n_clusters=7, n_init=7, max_iter=300, random_state=0).fit(data_set_scale)
        kmeans1.score(data_set_scale)
        prediction1 = kmeans1.predict(data_set_scale)

        kmeans2 = KMeans(n_clusters=7, n_init=7, max_iter=300, random_state=8).fit(data_set_scale)
        kmeans2.score(data_set_scale)
        prediction2 = kmeans2.predict(data_set_scale)

        confusion_matrix_kmeans = confusion_matrix(prediction1, prediction2)
        print(confusion_matrix_kmeans)

        heatmap(confusion_matrix_kmeans, annot=True, cmap='YlGn')
        title('Matrice de Confusion')
        ylabel('Véritable Label')
        xlabel('Label Théorique')
        show()

        kmeans3 = KMeans(n_clusters=7, n_init=7, max_iter=300, random_state=8).fit(data_set_scale)
        y_prediction = kmeans3.labels_
        kam = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        Y_num = []
        for j in B:
            Y_num.append(kam[j])
        score = silhouette_score(data_set_scale, y_prediction)
        print(score)

    def display_pca_components(self):
        fig = figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Les 2 premiers composants PCA')
        targets = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        colors = ['red', 'blue', 'green', 'purple', 'grey', 'orange', 'yellow']
        for target, color in zip(targets, colors):
            indice = self.concat_data_for_PCA["day"] == target
            ax.scatter(self.concat_data_for_PCA.loc[indice, 'PC1'],
                       self.concat_data_for_PCA.loc[indice, 'PC2'],
                       c=color, s=50)
        ax.legend(targets)
        ax.grid()
        show()