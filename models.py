from itertools import product
from math import sqrt

from matplotlib.pyplot import figure, ylabel, xlabel, title, plot, legend
from pandas import concat
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from main import Shealth


class StepsAggregatedByWeekModel:

    train_data = Shealth.week_train_data_set['steps']
    test_data = Shealth.week_test_data_set['steps']
    full_data = concat([train_data, test_data])
    p = range(0, 4, 1)
    d = 1
    q = range(0, 4, 1)
    P = range(0, 4, 1)
    D = 1
    Q = range(0, 4, 1)
    s = 4
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    # result_df = optimize_SARIMA(parameters_list, 1, 1, 4, train_data)
    # print(result_df)
    ad_fuller_result = adfuller(train_data) # adf stationary statistical test, p_value < 0.05 in our case
    kpss_test = kpss(train_data)  # kpss test stationary > 0.05 so its stationary
    auto_arima_steps_month = auto_arima(train_data, trace=False, seasonal=True)  # arima (0, 0, 3)

    order = (0, 0, 0)
    seasonal_order = (2, 0, 2, 18)
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model = model.fit(disp=False)
    summary = model.summary()

    pred = model.get_forecast(len(test_data))
    y_pred_df = pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = model.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test_data.index
    y_pred_out = y_pred_df["Predictions"]

    final_model = SARIMAX(full_data, order=order, seasonal_order=seasonal_order)  # use our model on full data
    final_model = final_model.fit()
    future_pred = final_model.predict(len(full_data), len(full_data) + 35)

    rmse = sqrt(mean_squared_error(test_data, y_pred_out))

    def get_acf_and_pacf_visualisation(self):
        """
        :return: the acf and pacf graph, useful for choosing arima coef
        """
        fig = figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.train_data.values.squeeze(), ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.train_data, ax=ax2)

    def display_model_evaluation_graph(self):
        self.train_data.plot()
        self.test_data.plot()
        ylabel('Steps')
        xlabel('Week')
        title("Train/Test split for steps prediction")
        plot(self.y_pred_out, color='green', label='Predictions')
        legend()

    def display_future_prediction(self):
        self.full_data.plot()
        self.future_pred.plot()
        ylabel('Steps')
        xlabel('Week')
        legend()

    def display_and_pacf_graph(self):
        fig = figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.train_data, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.train_data, ax=ax2)