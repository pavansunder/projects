# Shampoo sales forecasting for 5 years

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-ticks')

ts = pd.read_csv('datasets/shampoo_sales.csv')
ts.dropna(inplace = True)
print(ts.head())

ts['Month']=pd.to_datetime('200'+ts.Month,format='%Y-%m')
print(ts.head())

ts.rename(columns={"Sales of shampoo over a three year period": "Sales"}, inplace = True)
print(ts.head())

ts.set_index('Month', inplace = True)
print(ts.head())

fig, ax = plt.subplots(figsize = (10,6))
sb.lineplot(data = ts, ax = ax)
plt.show()

def test_stationary(timeseries):
    #Determing rolling statistics
    moving_avg = timeseries.rolling(window=12).mean()
    moving_std = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    fig, ax = plt.subplots(figsize = (10,4))
    sb.lineplot(data = timeseries, legend = False, label = 'Original')
    sb.lineplot(data = moving_avg, palette = ['red'], legend = False, label = 'Rollmean')
    sb.lineplot(data = moving_std, palette = ['black'], legend = False, label = 'Rollstd')
    plt.title('Rolling statistics to check stationarity')
    plt.legend(loc='best')
    plt.show()
    
    #Perform Dickey-Fuller test:
    from statsmodels.tsa.stattools import adfuller

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    print('Testing test statistic ({}) < 1% confidence interval ({})?: {}'.format(dftest[0], dftest[4]['1%'], (dftest[0] < dftest[4]['1%']) ))
    return dftest

stationarity = test_stationary(ts)

# data is non stationary, Therefore we have to make it stationary
# apply differencing technique to make the data stationary 
ts_log = np.log(ts)
data_diff = ts_log - ts_log.rolling(window = 12).mean()
data_diff.dropna(inplace = True)

stationarity = test_stationary(data_diff)

expwighted_avg = pd.DataFrame.ewm(ts_log, halflife=12).mean()
data_diff2 = ts_log - expwighted_avg
data_diff2.dropna(inplace = True)

stationarity = test_stationary(data_diff2)

ts_trans = np.sqrt(np.log(ts))
data_diff3 = ts_trans.diff()
data_diff3.dropna(inplace = True)

stationarity = test_stationary(data_diff3)

# we observe the data is stationary and can be used for prediction
# prediction is done using ARIMA

from pandas.plotting import autocorrelation_plot as acf
from statsmodels.tsa.stattools import pacf

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10,6))
fig.tight_layout(pad=4.0)
pacf_graph = pacf(ts_trans, nlags=20, method='ols')
sb.lineplot(data = pacf_graph, ax = ax[0])
ax[0].set_xticks(range(1,len(pacf_graph)))
ax[0].set_xlabel('Lag')
ax[0].set_ylabel('Partial auto correlation')
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(pacf_graph)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(pacf_graph)),linestyle='--',color='gray')
ax[0].set_title('Figure 4(a): Partial autocorrelation function')
acf_plot = acf(ts_trans, ax = ax[1])
ax[1].set_xticks(range(1,20))
ax[1].set_title('Figure 4(b) : Auto correlation funtion')
plt.show()

# based on acf and pacf plots the ARMA parameter can be p = 1, q = 6

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_trans, order=(1, 1, 6))
results_ARIMA = model.fit(disp = -1)
fig, ax = plt.subplots(figsize = (10,6))
sb.lineplot(data = data_diff3, ax = ax)
sb.lineplot(data = results_ARIMA.fittedvalues, ax = ax, legend = False, label = 'Fitted values' )
plt.title('ARIMA Model \n Residual sum of squares (RSS): %.4f'% sum((results_ARIMA.fittedvalues-data_diff3['Sales'])**2))
plt.legend(loc = 'best')
plt.show()

ARIMA_fittedvalues = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(ARIMA_fittedvalues.head())

predictions_ARIMA_cumsum = ARIMA_fittedvalues.cumsum()
print(predictions_ARIMA_cumsum.head())

predictions_ARIMA = pd.Series(ts_trans['Sales'], index=ts_trans.index)
predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_cumsum,fill_value=0)
print(predictions_ARIMA.head())

predictions_ARIMA = (np.exp(predictions_ARIMA))**2
figure, ax = plt.subplots(figsize = (10,6))
sb.lineplot(data = ts, ax = ax)
sb.lineplot(data = predictions_ARIMA, ax = ax, palette = ['red'], legend = False, label = 'Fitted values back to original')
plt.title('Fitted values to original form')
plt.legend(loc = 'best')

ts.info()

print('''we have 36 data points.\n
if we have to predict for next 5 yrears ie 5 * 12 months = 60\n 
So, we have to add 60 to our original length of data ie {}'''.format(36 + 60))

figure, ax = plt.subplots(figsize = (10,6))
results_ARIMA.plot_predict(1,96, ax = ax)
plt.title('5 years forecast')
plt.show()

