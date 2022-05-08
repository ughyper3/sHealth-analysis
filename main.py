from math import sqrt, log, exp
from pandas import read_csv, DataFrame
from process import ProcessDataset
from seaborn import pairplot, lineplot, heatmap

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import cophenet, inconsistent , maxRstat
from scipy.spatial.distance import pdist


#from seaborn import sns, set #marche pas, à essayer sur vos pc please
import warnings
warnings.filterwarnings('ignore') #pour enlever les warnings du doc car c'était pas très beau


class Shealth:

    process = ProcessDataset()
    raw_data = read_csv('shealth.csv',  sep=";", decimal=',')
    data_set = process.process_data(raw_data)

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
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns; sns.set()
    from scipy.stats import chi2_contingency


    def display_heat_map(self):
        """
<<<<<<< Updated upstream
        :return: Graph of correlation
        """      
        return heatmap(self.correlation, annot=True).plot()
=======
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
        
    #il comprend pas le sns.. pourtant la librairie est bien installée sur mon pc, à essayer chez vous svp    
    #def display_histograms(self):
        #sns.distplot(data_set['steps'], bins=30, color='green', ax = axes[0,0]) \
            #.set(title='Histo', xlabel='to define', ylabel='to define');
        #sns.distplot(data_set['sport_duration'], bins=30, color='red', ax = axes[0,1]) \
            #.set(title='Histo', xlabel='to define', ylabel='to define');
        #sns.distplot(data_set['spent_energy'], bins=30, color='blue', ax = axes[1,0]) \
            #.set(title='Histo', xlabel='to define', ylabel='to define');
        #sns.distplot(data_set['average_speed'], bins=30, color='yellow', ax = axes[0,1]) \
            #.set(title='Histo', xlabel='to define', ylabel='to define');

            
    #def display_mds_method(self):
        #data_normalized = scaler.fit_transform(data_set)
        #mds = MDS(n_components=2, random_state=0)
        #df_mds = mds.fit_transform(data_normalized)
        #fig, ax = plt.subplots(figsize=(20, 16))
        #sns.scatterplot(x=df_mds[:, 0], y=df_mds[:, 1], hue=data_set['day'])
        #plt.axis('Equal')
        #ax.set_title('Variable factor map')
    
    
    #Lab 6, faut que je revois je capte po l'erreur alors que c est défini 
    def display_truncated_method(self):
        #display truncated dendrogram
        Z = linkage(data_set, 'ward', optimal_ordering=True)
        c , coph_dists = cophenet (Z , pdist(data_set))
        
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        dendrogram (
            Z,
            truncate_mode='lastp', 
            p=12, 
            show_leaf_counts=False,
            leaf_rotation=90.,
            leaf_font_size =12.,
            show_contracted=True,
        )
        plt.show()
>>>>>>> Stashed changes
