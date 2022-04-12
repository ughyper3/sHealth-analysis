from math import sqrt, log, exp
from pandas import read_csv, DataFrame
from process import ProcessDateset
from matplotlib.pyplot import plot, show


class Shealth:

    process = ProcessDateset()

    raw_data = read_csv('shealth.csv',  sep=";", decimal=',', parse_dates=['Date'])
    data_set = process.process_data(raw_data)

    data_set_description = data_set.describe()
    day = data_set.day
    date = data_set.date
    weight = data_set.weight
    steps = data_set.steps
    walk_duration = data_set.walk_duration
    sport_duration = data_set.sport_duration
    spent_energy = data_set.spent_energy
    country_number = data_set.country_number


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

    def show_graph_time_steps(self):
        """
        Graph of the number of steps by time
        :return: void
        """
        plot(self.date, self.steps, 'ro')
        show()


