from numpy import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pandas import Series


class ArmaModels:

    @staticmethod
    def fit_ARMA_model(train_data_set: Series, test_data_set: Series, attribute: str, order: tuple, alpha: float):#
        """
        this method use arma model
        :param train_data_set: dataset frame that we use for training
        :param test_data_set: dataset frame that we use for testing
        :param attribute: dataset attribute that we want to predict the evolution
        :param order: cf arma model doc https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
        :param alpha:
        :return: the prediction and a rsme measure of error
        """
        y_train = train_data_set[attribute]
        y_test = test_data_set[attribute]

        ARMAmodel = SARIMAX(y_train, order=order)
        ARMAmodel = ARMAmodel.fit(disp=False)

        y_pred = ARMAmodel.get_forecast(len(test_data_set.index))
        y_pred_df = y_pred.conf_int(alpha=alpha)
        y_pred_df["Predictions"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df.index = test_data_set.index
        y_pred_out = y_pred_df["Predictions"]
        
        arma_rmse = sqrt(mean_squared_error(y_test.values, y_pred_df["Predictions"]))
        return y_pred_out, arma_rmse


class ArimaModels:

    @staticmethod
    def fit_ARIMA_model(train_data_set: Series, test_data_set: Series, attribute: str, order: tuple, alpha: float):
        """
        this method use arima model
        :param train_data_set: dataset frame that we use for training
        :param test_data_set: dataset frame that we use for testing
        :param attribute: dataset attribute that we want to predict the evolution
        :param order: cf arma model doc https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
        :param alpha:
        :return: the prediction and a rsme measure of error
              """
        y_train = train_data_set[attribute]
        y_test = test_data_set[attribute]

        ARIMAmodel = ARIMA(y_train, order=order)
        ARIMAmodel = ARIMAmodel.fit()

        y_pred = ARIMAmodel.get_forecast(len(test_data_set.index))
        y_pred_df = y_pred.conf_int(alpha=alpha)
        y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df.index = test_data_set.index
        y_pred_out = y_pred_df["Predictions"]

        arima_rmse = sqrt(mean_squared_error(y_test.values, y_pred_df["Predictions"]))
        return y_pred_out, arima_rmse


class SarimaModels:

    @staticmethod
    def fit_SARIMA_model(train_data_set: Series, test_data_set: Series, attribute: str, order: tuple, seasonal_order: tuple, alpha: float):
        """
        this method use sarima model
        :param train_data_set: dataset frame that we use for training
        :param test_data_set: dataset frame that we use for testing
        :param attribute: dataset attribute that we want to predict the evolution
        :param order: cf arma model doc https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
        :param seasonal_order: cf doc
        :param alpha:
        :return: the prediction and a rsme measure of error
              """
        y_train = train_data_set[attribute]
        y_test = test_data_set[attribute]

        SARIMAXmodel = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
        SARIMAXmodel = SARIMAXmodel.fit(disp=False)

        y_pred = SARIMAXmodel.get_forecast(len(test_data_set.index))
        y_pred_df = y_pred.conf_int(alpha=alpha)
        y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df.index = test_data_set.index
        y_pred_out = y_pred_df["Predictions"]

        sarima_rmse = sqrt(mean_squared_error(y_test.values, y_pred_df["Predictions"]))
        return y_pred_out, sarima_rmse