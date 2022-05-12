from pandas import DataFrame
from statsmodels.tsa.statespace.sarimax import SARIMAX


def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
        medium article : https://towardsdatascience.com/time-series-forecasting-with-sarima-in-python-cda5b793977b
    """

    results = []

    for param in parameters_list:
        try:
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue

        aic = model.aic
        results.append([param, aic])

    result_df = DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df