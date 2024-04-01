# -------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      EPervago
#
# Created:     11/11/2022
# Copyright:   (c) EPervago 2022
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import skgstat as skg
import openturns as ot
from pprint import pprint
from skgstat.plotting import backend
from scipy import stats
import random
from Bivariate_analyses import get_label
from ScatterD import plot_scatter_with_distrib


# String identifying the semi-variance estimator to be used. Defaults to the Matheron estimator. Possible values are:
estimator_list = {
    "matheron": "Matheron",  # default
    "cressie": "Cressie-Hawkins",
    "dowd": "Dowd-Estimator",
    "genton": "Genton",
    "minmax": "MinMax Scaler",
    "entropy": "Shannon Entropy",
}

# String identifying the theoretical variogram function to be used to describe the experimental variogram. Can be one of:
model_list = {
    "spherical": "Spherical",  # default]
    "exponential": "Exponential",
    "gaussian": "Gaussian",
    "cubic": "Cubic",
    "stable": "Stable model",
    "matern": "Matérn model",
    "nugget": "nugget effect variogram",
}

def Trend_Analyses_1D(sample, units, width,height,
        ntrend=0, removetrend=False):

    # Trend Analysis 1D
    # Input:
    #   sample - sample with 2 columns : depth, vals
    #   width,height - windows width and height
    #   ntrend - integer : 0,1,2. Default 0. Trend order.
    #   removetrend - boolean. Default False. If True, before all calculating trend will be removed.
    # Output:
    #   results : dictionary:
    #    plots:
    #       "Trend" : Fig_Trend
    #    vals : nparray, detrended values

    results = {}
    var_names = sample.getDescription()
    depth = np.array(sample[:,0]).flatten()
    vals = np.array(sample[:,1]).flatten()
    label_0 = get_label(var_names[0], units)
    label_1 = get_label(var_names[1], units)

    Fig_Trend, ax = plt.subplots()
    dpi = Fig_Trend.get_dpi()
    new_size = (width / dpi, height / dpi)
    Fig_Trend.set_size_inches(new_size)

    # V.location_trend(axes=ax, show=False, add_trend_line = True)
    ax.plot(depth, vals, "b-", label="Obsereved")
    ax.axhline(vals.mean(), ls="--", color="k", lw=3, label="Mean")
    ax.set_xlabel(label_0)
    ax.set_ylabel(label_1)
    ax.grid(which="major")
    # ax[1].set_visible(False)
    N = len(depth)
    dmin = depth.min()
    dmax = depth.max()
    if ntrend > 0:
        if ntrend == 1:
            W = np.vstack((np.ones(N), depth)).T
            C = np.linalg.lstsq(W, vals, rcond=None)[0]
            ax.axline(
                (dmin, C[0] + C[1] * dmin),
                (dmax, C[0] + C[1] * dmax),
                linewidth=3,
                color="r",
                ls="--",
                label="Linear trend",
            )
        elif ntrend == 2:
            W = np.vstack((np.ones(N), depth, np.square(depth))).T
            C = np.linalg.lstsq(W, vals, rcond=None)[0]
            d = np.linspace(dmin, dmax, 101)
            t = C[0] + C[1] * d + C[2] * np.square(d)
            ax.plot(d, t, linewidth=3, color="r", ls="--", label="Quadratic trend")
        else:
            raise Exception(f"Wrong order of trend: {ntrend}")
    if removetrend and ntrend > 0:
        tr = np.matmul(W, C)
        vals_tr = vals - tr
        ax.plot(depth, vals_tr, "g-", label="Detrended")
        sample_tr = ot.Sample(sample)
        s_add = ot.Sample(vals_tr.reshape((N, 1)))
        sample_tr[:, 1] = s_add
        names = list(sample_tr.getDescription())
        names[1] += " TR"
        sample_tr.setDescription(names)
        results["sample_tr"] = sample_tr
        vals = vals_tr
    ax.legend(loc="best")
    results["Trend"] = Fig_Trend
    return results, vals


def Variogram_Analysis(
    sample,
    units,
    width,
    height,
    n_lags=15,
    estimator="matheron",
    model="spherical",
    bin_func="even",
    ntrend=0,
    removetrend=False,
    valid_full=False,
):
    # Input:
    #   sample - sample with 2 columns, first - depth
    #   width,height - windows width and height
    #   n_lags -  Specify the number of lag classes to be defined by the binning function.
    #   estimator - String identifying the semi-variance estimator to be used. Defaults to the Matheron estimator.
    #           Possible values are: "matheron"(default), "cressie", "dowd", "genton", "minmax", "entropy" (estimator_list.keys())
    #   model - String identifying the theoretical variogram function to be used to describe the experimental variogram.
    #           Can be one of: "spherical"(default), "exponential", "gaussian", "cubic", "stable", "matern", "nugget"
    #   ntrend - integer, order of trend. Default 0. Can be 0,1,2.
    #   bin_func - String identifying the binning function used to find lag class
    #        edges. All methods calculate bin edges on the interval [0, maxlag[.
    #        Possible values are:
    #            * 'even'` (default) finds `n_lags` same width bins
    #            * 'uniform'` forms `n_lags` bins of same data count
    #            * 'fd'` applies Freedman-Diaconis estimator to find `n_lags`
    #            * 'sturges'` applies Sturge's rule to find `n_lags`.
    #            * 'scott'` applies Scott's rule to find `n_lags`
    #            * 'doane'` applies Doane's extension to Sturge's rule
    #                   to find `n_lags`
    #            * 'sqrt' uses the square-root of
    #                       :func:`distance <skgstat.Variogram.distance>`
    #                       as `n_lags`.
    #            * 'kmeans' uses KMeans clustering to well supported bins
    #            * 'ward' uses hierachical clustering to find
    #                   minimum-variance clusters.

    #   removetrend - boolean. Default False. If True, before all calculating trend will be removed.
    # Output: dictionary
    #   "Variogram" : skgstat.Variogram
    #   "Statistics" : statistics table, Pandas Dataframe
    #   "RMSE" - RMSE
    #   "BIC" - BIC - deleted
    #   "AIC" - AIC - deleted
    #   "range"
    #   "sill"
    #   "nugget"
    #    plots:
    #       "Trend" : Fig_Trend
    #       "VariogramEmp" : Fig_VariogramEmp
    #       "VariogramFitting" : Fig_VariogramFitting
    #       "CrossValidation" : Fig_CrossValidation
    results = {}
    var_names = tuple(sample.getDescription())
    depth = np.array(sample[:, 0]).flatten()
    #vals = np.array(sample[:, 1]).flatten()
    label_0 = get_label(var_names[0], units)
    label_1 = get_label(var_names[1], units)

    res, vals = Trend_Analyses_1D(sample, units, width,height,ntrend,removetrend)

    V = skg.Variogram(
        depth,
        vals,
        maxlag="median",
        n_lags=n_lags,
        normalize=False,
        estimator=estimator,
        model=model,
        bin_func=bin_func,
    )
    results["Variogram"] = V
    results["RMSE"] = V.rmse
    # results["BIC"] = V.bic - deletes
    # results["AIC"] = V.aic - deleted
    results["range"] = V.parameters[0]
    results["sill"] = V.parameters[1]
    if len(V.parameters) == 3:
        results["nugget"] = V.parameters[2]
    else:
        results["nugget"] = V.parameters[3]

    # Number of pairs per lag bin
    pairs_count = np.fromiter((g.size for g in V.lag_classes()), dtype=int)

    # Empirical variogram

    Fig_VariogramEmp, ax = plt.subplots(1, 1)
    dpi = Fig_VariogramEmp.get_dpi()
    new_size = (width / dpi, height / dpi)
    Fig_VariogramEmp.set_size_inches(new_size)

    ax.plot(V.bins, V.experimental, ".-", markersize=12, label="Empirical variogram",
        color="cyan",
        markerfacecolor = "blue",
        alpha=0.8,
        lw=4,
    )

    for x, y, p in zip(V.bins, V.experimental, pairs_count):
        label = "{:.0f}".format(p)

        plt.annotate(
            label,  # this is the text
            (x, y),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(10, 0),  # distance from text to points (x,y)
            color="b",
            ha="left",
            va="center",
        )  # horizontal alignment can be left, right or center
    ax.axhline(vals.var(), ls="--", color="r", lw=2, label="Variance")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Semivariance (%s)" % estimator_list[estimator])
    ax.grid(which="major")
    ax.legend(loc="best")
    results["VariogramEmp"] = Fig_VariogramEmp
    # plt.show()

    # Variogram Fitting
    Fig_VariogramFitting, ax = plt.subplots(1, 1)
    Fig_VariogramFitting.set_size_inches(new_size)
    V.plot(axes=ax, hist=True, grid=False, show=False)
    ax.get_children()[0].set(markersize=12, marker='.')
    ax.get_children()[1].set(color="r", alpha=0.8,linewidth=4)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Semivariance (%s)" % estimator_list[estimator])
    ax.grid(which="major")
    ax.set_title(
        "Model: %s; RMSE: %.2e" % (model_list[V.describe()["model"]], V.rmse),
        fontsize=12,
    )
    results["VariogramFitting"] = Fig_VariogramFitting
    # plt.show()

    # Number of pairs per lag bin
    pairs_count = np.fromiter((g.size for g in V.lag_classes()), dtype=int)
    ##    print('--------------------------------------------------------')
    ##    print('N Pairs      Lags          Semivariances')
    ##    print('--------------------------------------------------------')
    ##    for i in range(n_lags):
    ##        print(i+1,' ', pairs_count[i],' ', V.distance[i],' ', V.experimental[i] )
    ##    print('--------------------------------------------------------')
    ##    print('')
    d = {
        "Lags": list(range(1, n_lags + 1)),
        "Pairs": pairs_count[:n_lags],
        "Distance": V.distance[:n_lags],
        "Semivariances": V.experimental,
    }
    stat_table = pd.DataFrame(d)
    results["Statistics"] = stat_table

    # The describe method will return the most important parameters as a dictionary.
    ##    print('--------Variogram description-------')
    ##    pprint(V.describe())
    ##
    ##    print(V)

    # Fig_DiffPlot, ax = plt.subplots(1,1)
    # V.distance_difference_plot()

    # Cross Validation (jacknife, leave one out)
    """
    Leave-one-out cross validation of the given variogram model using the
    OrdinaryKriging instance. This method can be called using Variogram.cross_validate.
    """
    # RMSE = V.cross_validate(method = 'jacknife', n = None, metric = 'rmse', seed=None)
    # print('----------------------------------------------------')
    # print('Variogram Cross Validation RMSE = ', RMSE)
    # print('----------------------------------------------------')
    N = len(depth)
    NS = 400  # Number of sample for cross validation calculation
    NS = min(NS, N)
    if valid_full or NS >= N:
        index = range(N)
        Z0 = vals
    else:
        index = random.sample(range(N), NS)
        Z0 = vals[index]

    Z = np.full_like(Z0, None)
    depth = depth.reshape((N, 1))
    for idx, k in zip(index, range(NS)):
        c = np.delete(depth, idx, axis=0)
        v = np.delete(vals, idx, axis=0)
        ok = skg.OrdinaryKriging(V, coordinates=c, values=v)

        # interpolate Z[idx]
        Z[k] = ok.transform(depth[idx])

    label_1a = get_label(var_names[1], units, "*")
    Fig_CrossValidation = plot_scatter_with_distrib(
        Z0, Z, namex=label_1, namey=label_1a, size=(width, height)
    )
    axd = Fig_CrossValidation.get_axes()
    vmin = vals.min()
    vmax = vals.max()
    axd[3].axline((vmin, vmin), (vmax, vmax), linewidth=1, color="r")
    results["CrossValidation"] = Fig_CrossValidation

    return results


def main():
    os.chdir(
        "C:\\Users\\EPervago\\Documents\\Python\\Copula cosimalacion\\Copulas Bernstein 01-10-2022\\Python"
    )
    Sample = ot.Sample.ImportFromCSVFile("Z32.csv", ",")
    sample = Sample[:, [0, 9]]
    N = sample.getSize()
    units = {sample.getDescription()[0]: "m", sample.getDescription()[1]: "vol"}
##    res, vals = Trend_Analyses_1D(sample, units, 700,700,ntrend=2,removetrend=True)
##    plt.show()
    result = Variogram_Analysis(
        sample, units, 700, 700, n_lags=30, bin_func="even", ntrend=1, removetrend=True
    )
    plt.show()


if __name__ == "__main__":
    main()
