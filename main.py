from math import sqrt, log, exp

from matplotlib.pyplot import subplots
from pandas import read_csv, DataFrame
from process import ProcessDataset
from seaborn import pairplot, lineplot, pointplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Shealth:

    process = ProcessDataset()
    raw_data = read_csv('shealth.csv',  sep=";", decimal=',')
    data_set = process.process_data(raw_data)

    data_set_without_categorical_variables = data_set.drop(['day', 'date', 'country_number', 'week'], axis=1)
    scaler = StandardScaler()
    data_set_scale = scaler.fit_transform(data_set_without_categorical_variables)
    n_components = 4
    pca = PCA(n_components=n_components)
    data_set_without_categorical_variables_proj = pca.fit_transform(data_set_scale)
    variance_of_first_components = pca.explained_variance_ratio_


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
