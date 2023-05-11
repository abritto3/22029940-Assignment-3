import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import itertools as iter


def scaler(df):
    """ Expects a dataframe and normalises all 
        columns to the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    """
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


df = pd.read_excel('CO2.xlsx')

#Clean the dataset but removing the rows and columns containing Nan or inf values
df_ex = df.iloc[0:265, 34:-3]
df_ex = df_ex.drop([3,9,21,27,36,39,50,65,69,75,77,79,86,94,105,117,130,136,140,146,158,164,172,183,200,212,213,252],axis = 0)

#Normalise the dataset
df_norm, df_min, df_max = scaler(df_ex)
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = np.array(cen)

#Find the axis cluster centres
xcen = cen[:, 0]
ycen = cen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm[1990], df_norm[2015], 80, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("CO2 emissions (1990)")
plt.ylabel("CO2 emissions (2019)")
plt.title("CO2 Emission comparision 1990 X 2015")
plt.savefig('cluster.png', format='png', dpi=300)
plt.show()

#Backscale the normalised cluster
cen = backscale(cen, df_min, df_max)
xcen = cen[:, 0]
ycen = cen[:, 1]

#Transpose the cleaned dataset for fitting
df_co4 = df_ex.iloc[0:10].T
df_co4.index = pd.to_numeric(df_co4.index)

#Plot the dataset to guess the p0 values
plt.plot(df_co4.index, df_co4[0], marker = "o")
plt.show()

#Find the optimal and covariance value using curve_fit
param, covar = curve_fit(logistic, df_co4.index, df_co4[0], p0=(0.5e5, 0.03, 2005.0))

#plot the logistic function with the dataset
sigma = np.sqrt(np.diag(covar))
df_co4["fit"] = logistic(df_co4.index, *param)
plt.plot(df_co4.index, df_co4[0], label = "CO2")
plt.plot(df_co4.index, df_co4["fit"], label = "fit")
plt.show()

#Display the turning points and growth rate
print("Tsurning point", param[2], "+/-", sigma[2])
print("CO2 emissions at turning point", param[0]/1e9, "+/-", sigma[0]/1e9)
print("Growth rate", param[1], "+/-", sigma[1])

#Predict the data for 2025 using the logistical function
year = np.arange(1990, 2025)
forecast = logistic(year, *param)

#Find the error ranges
low, up = err_ranges(year, logistic, param, sigma)

#Plot the dataset, logistical function with the prediction for 2025 and the error ranges together
plt.figure()
plt.plot(df_co4.index, df_co4[0], label="CO2")
plt.plot(year, forecast, label="Forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.4)
plt.xlabel("Year")
plt.ylabel("CO2 emissions (kt)")
plt.xlim(1990, 2025)
plt.legend(loc = "upper left")
plt.title("CO2 Emission rate and forecast for 2025")
plt.savefig('fitting.png', format='png', dpi=300)
plt.show()