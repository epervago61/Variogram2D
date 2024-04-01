# -------------------------------------------------------------------------------
# Name:        Bivariate_analyses
# Purpose:
#
# Author:      Evgeny Pervago epervago@imp.mx
#
# Created:     04/11/2022
# Update:      11/03/2023
# Copyright:   (c) Evgeny Pervago 2022
# Licence:     <your licence>
# -------------------------------------------------------------------------------
# ================================================================================================
# def bivariate_par(sample,marginals,copulas,copula_name,units,width,height,par={"color1":"blue","color2":"red"}):
# ------------------------------------------------------------------------------------------------
# Case parametric
# Input:
#   sample_2v - sample with 2 variables
#   marginals - list: marginal distribution[2]
#   copulas - dictionary: key = name of copula, copulas[key] = copula factory
#   copula_Name - string
#       "BestBIC" - search best kernel with BIC test
#       "BestAIC" - search best kernel with AIC test
#       <name> - fixed copula name
#   units - dictionary : variable name:variable units
#   width, height - figure size
#   par - colors
# Output: dictionary
#   "distribution" : best kernel smoothing distribution
#   test results:
#   "test_aic" : test_aic
#   "test_bic" : test_bic
#   "Simulation" : result of simulation
#   plots:
#   "Scatter"  : data scatter plot
#   "SimScatter"  : view_Scatter
#   "RankScatter" : view_RankScatter
#   "PDF"      : view_PDF
#   "CDF"      : view_CDF
#   "RegrXY"   : Quantile Regression X vs Y
#   "RegrDY"   : Quantile Regression Depth vs Y
# ================================================================================================
# def bivariate_ks(sample,marginals,kernels, kernel_name, units, width,height,par={"color1":"blue","color2":"red"}):
# ------------------------------------------------------------------------------------------------
# Case no parametric
# Input:
#   sample - sample with 2 variables
#   marginals - list: marginal distribution[2]
#   kernels - dictionary: key = name of kernel, kernels[key] = kernel distribution
#   kernel_Name - string
#       "BestBIC" - search best kernel with BIC test
#       "BestAIC" - search best kernel with AIC test
#       <name> - fixed kernel name
#   units - dictionary : variable name:variable units
#   width, height - figure size
#   par - colors
# Output: dictionary
#   "distribution" : best kernel smoothing distribution
#   test results:
#   "test_aic" : test_aic
#   "test_bic" : test_bic
#   "Simulation" : result of simulation
#   plots:
#   "Scatter"  : data scatter plot
#   "SimScatter"  : view_Scatter
#   "RankScatter" : view_RankScatter
#   "PDF"      : view_PDF
#   "CDF"      : view_CDF
#   "RegrXY"   : Quantile Regression X vs Y
#   "RegrDY"   : Quantile Regression Depth vs Y
# ================================================================================================
# def QuantileRegression(coord,sample,distribution,title,units,width,height):
# ------------------------------------------------------------------------------------------------
# ------ Quantile Regression------------------------------------------
# Input:
#   coord - ot.Sample, 1 var for 1D, 2 var for 2D
#   sample:
#       validation -> ot.Sample 2 var
#       prediction -> tuple(ot.Sample 1 var, pred name: string)
#   distribution - distribution
#   title - title, string
#   units - dictionary : variable name:variable units
#   width, height - figure size
#   interpolate:logical - True or False
#   colmap - Colormap
# Output:
#   "Fig1" - View, if "prediction" -> None
#   "Fig2" - View
#   "Fig2a","Fig2b","Fig2b", in case 2D.  If "prediction" -> "Fig2A"==None
#   "Data" - predicted or validated, ot.Sample, 1 var
# ================================================================================================

import openturns as ot
import openturns.viewer as otv
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import sys
import pandas as pd
import numpy as np
from matplotlib.markers import MarkerStyle
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from test_distribution import computeRMSE
from ScatterD import plot_scatter_with_distrib
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"font.size": 12})

warnings.filterwarnings("ignore")
ot.Log.Show(ot.Log.NONE)


# ================================================================================================
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad="2%")
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

# ================================================================================================
def getKernels():
    res = {}
    res["Normal"] = ot.Normal()
    res["Triangular"] = ot.Triangular()
    res["Epanechnikov"] = ot.Epanechnikov()
    res["Uniform"] = ot.Uniform()
    return res


# ================================================================================================
def getBestCopula_BIC_KS(kernels, marginals, sample):
    # Input:
    #   kernels: dictionary - kernel  name: kernel
    #   marginals: list 2 elements, marginal distributions
    #   sample: data
    # Output:
    #   best kernel, BIC
    print("Estimation non parametric copula by BIC:")
    best_bic = sys.float_info.max
    best_kernel = None
    best_kernel_name = None
    best_ks_copula = None
    for kname, kern in kernels.items():
        try:
            # Fit a non parametric copula using KernelSmoothing
            ks_copula = ot.KernelSmoothing(kern).build(sample).getCopula()
            distrib = ot.ComposedDistribution(marginals, ks_copula)
            #n_parameters = distrib.getParameter().getDimension()
            n_parameters = 0
            bic = ot.FittingTest.BIC(sample, distrib, n_parameters)
            if bic == sys.float_info.max:
                print("{:15}: *****".format(kname))
            else:
                print("{:15}: {:.2f}".format(kname, bic))
            if bic < best_bic:
                best_bic = bic
                best_kernel = kern
                best_kernel_name = kname
                best_ks_copula = ks_copula
        except:
            print("{:15}: *****".format(kname))
    if not best_ks_copula:
        best_bic = None
    return best_ks_copula, best_kernel, best_kernel_name, best_bic


# ================================================================================================
def getBestCopula_AIC_KS(kernels, marginals, sample):
    # Input:
    #   kernels: dictionary - kernel  name: kernel
    #   marginals: list 2 elements, marginal distributions
    #   sample: data
    # Output:
    #   best kernel, AIC
    print("Estimation non parametric copula by AIC:")
    best_aic = sys.float_info.max
    best_kernel = None
    best_ks_copula = None
    best_kernel_name = None
    for kname, kern in kernels.items():
        try:
            # Fit a non parametric copula using KernelSmoothing
            ks_copula = ot.KernelSmoothing(kern).build(sample).getCopula()
            distrib = ot.ComposedDistribution(marginals, ks_copula)
            #n_parameters = distrib.getParameter().getDimension()
            n_parameters = 0
            aic = ot.FittingTest.AIC(sample, distrib, n_parameters)
            if aic == sys.float_info.max:
                print("{:15}: *****".format(kname))
            else:
                print("{:15}: {:.2f}".format(kname, aic))
            if aic < best_aic:
                best_aic = aic
                best_kernel = kern
                best_kernel_name = kname
                best_ks_copula = ks_copula
        except:
            print("{:15}: *****".format(kname))
    if not best_ks_copula:
        best_aic = None
    return best_ks_copula, best_kernel, best_kernel_name, best_aic


# ================================================================================================
def getCopulaFactoriesList():
    # List of copula factories
    copulas = {}
    for factory in ot.DistributionFactory.GetContinuousMultiVariateFactories():
        if not factory.build().isCopula():
            continue
        if factory.getImplementation().getClassName() == "BernsteinCopulaFactory":
            continue
        copulas[str(factory).replace("CopulaFactory", "")] = factory.getImplementation()
    return copulas


# ================================================================================================
def getUnivariateFactoriesList():
    # List of univariate factories
    distr = {}
    for factory in ot.DistributionFactory.GetContinuousUniVariateFactories():
        if factory.getImplementation().getClassName() == "HistogramFactory":
            continue
        distr[str(factory).replace("Factory", "")] = factory.getImplementation()
    return distr


# ================================================================================================
def getBestCopula_BIC(copulas, marginals, sample):
    # Input:
    #   copulas: dictionary - copula factory name: copula factory
    #   marginals: list 2 elements, marginal distributions
    #   sample: data
    # Output:
    #   best copula, BIC
    #       OR
    #  None, None
    best_bic = sys.float_info.max
    best_copula = None
    best_copula_name = None
    print("Estimation parametric copula by BIC:")
    for cname, cop in copulas.items():
        try:
            estimated_copula = cop.build(sample)
            distrib = ot.ComposedDistribution(marginals, estimated_copula)
            Np = distrib.getParameter().getDimension()
            bic = ot.FittingTest.BIC(sample, distrib, Np)
            if bic == sys.float_info.max:
                print("{:15}: *****".format(cname))
            else:
                print("{:15}: {:.2f}".format(cname, bic))
            # if bic < 0.0:
            #    continue
            if bic < best_bic:
                best_bic = bic
                best_copula = estimated_copula
                best_copula_name = cname
        except:
            print("{:15}: *****".format(cname))
    if not best_copula:
        best_bic = None
    return best_copula, best_copula_name, best_bic


# ================================================================================================
def getBestCopula_AIC(copulas, marginals, sample):
    # Input:
    #   copulas: dictionary - copula factory name: copula factory
    #   marginals: list 2 elements, marginal distributions
    #   sample: data
    # Output:
    #   best copula, AIC
    #       OR
    #  None, None
    best_aic = sys.float_info.max
    best_copula = None
    best_copula_name = None
    print("Estimation parametric copula by AIC:")
    for cname, cop in copulas.items():
        try:
            estimated_copula = cop.build(sample)
            distrib = ot.ComposedDistribution(marginals, estimated_copula)
            Np = distrib.getParameter().getDimension()
            aic = ot.FittingTest.AIC(sample, distrib, Np)
            if aic == sys.float_info.max:
                print("{:15}: *****".format(cname))
            else:
                print("{:15}: {:.2f}".format(cname, aic))
            # if bic < 0.0:
            #    continue
            if aic < best_aic:
                best_aic = aic
                best_copula = estimated_copula
                best_copula_name = cname
        except:
            print("{:15}: *****".format(cname))
    if not best_copula:
        best_aic = None
    return best_copula, best_copula_name, best_aic


# ================================================================================================
def getUnivariateStatistics(sample, suffix=""):
    # Input sample - OpenTurns Sample
    # Output - Pandas DataFrame
    names = [
        "Size",
        "Name",
        "Minimum",
        "1st quartile",
        "Mean",
        "Median" "3rd quartile",
        "Maximum",
        "Range",
        "Variance",
        "Standard Deviation",
        "Skewness",
        "Kurtosis",
    ]
    var_names = list(sample.getDescription())
    if suffix:
        var_names = [s + suffix for s in var_names]
    data = np.zeros((12, 2))
    result = pd.DataFrame(data, index=names, columns=var_names)
    result.loc["Size"] = [sample.getSize()] * 2
    result.loc["Name"] = list(sample.getDescription())
    result.loc["Minimum"] = list(sample.getMin())
    result.loc["1st quartile"] = list(sample.computeQuantilePerComponent(0.25))
    result.loc["Mean"] = list(sample.computeMean())
    result.loc["Median"] = list(sample.computeMedian())
    result.loc["3rd quartile"] = list(sample.computeQuantilePerComponent(0.75))
    result.loc["Maximum"] = list(sample.getMax())
    result.loc["Range"] = list(sample.computeRange())
    result.loc["Variance"] = list(sample.computeVariance())
    result.loc["Standard Deviation"] = list(sample.computeStandardDeviation())
    result.loc["Skewness"] = list(sample.computeSkewness())
    result.loc["Kurtosis"] = list(sample.computeKurtosis())
    return result


# ================================================================================================
def getBivariateStatistics(sample, col_name):
    # Input sample - OpenTurns Sample with 2 variables
    # Output - Pandas DataFrame
    names = ["Pearson correlation", "Kendall correlation", "Spearman correlation"]
    data = np.zeros((3, 1))
    result = pd.DataFrame(data, index=names, columns=(col_name,))

    result.loc["Pearson correlation"] = sample.computePearsonCorrelation()[0, 1]
    result.loc["Kendall correlation"] = sample.computeKendallTau()[0, 1]
    result.loc["Spearman correlation"] = sample.computeSpearmanCorrelation()[0, 1]
    return result


# ================================================================================================
def get_label(var_name, units, suff=""):
    label = var_name + suff
    if var_name in units:
        label += " [" + units.get(var_name,'') + "]"
    return label
# ================================================================================================
def plot_QuantileRegression(sample, distribution, title, units, width, height):

    # Estimate a conditional quantile
    if isinstance(sample, (list, tuple)):
        pvar_name = sample[1]
        sample = sample[0]
        var_names = tuple(sample.getDescription())
        var_names = var_names + (pvar_name,)
    else:
        var_names = tuple(sample.getDescription())

    if sample.getDimension()==1:
        x = sample[:, 0]
        is_validation = False
    elif sample.getDimension()==2:
        x = sample[:, 0]
        y = sample[:, 1]
        is_validation = True

    nc = 100

    if is_validation:
        y0 = np.array(y)
        ym = [distribution.computeConditionalQuantile(0.5, xi) for xi in x]
        ym = np.reshape(ym, y0.shape)

        RMSE = np.sqrt(np.square(np.subtract(y0, ym)).mean())

    X = np.linspace(x.getMin(), x.getMax(), nc)

    Y_Q1 = [distribution.computeConditionalQuantile(0.25, xi) for xi in X]
    Y_Q2 = [distribution.computeConditionalQuantile(0.5, xi) for xi in X]
    Y_Q3 = [distribution.computeConditionalQuantile(0.75, xi) for xi in X]

    X = X.flatten().tolist()

    label_0 = get_label(var_names[0], units)
    label_1 = get_label(var_names[1], units)

    if is_validation:
        full_title = "Conditional Quantile Regression with " + \
                        title + \
                        " of {1} | {0}\nRMSE = {2:.3g}".format(*var_names, RMSE)
    else:
        full_title = "Conditional Quantile Regression with " + \
                        title + \
                        " of {1} | {0}".format(*var_names)

    fig, ax = plt.subplots(1, 1)
    dpi = fig.get_dpi()
    new_size = (width / dpi, height / dpi)
    fig.set_size_inches(new_size)
    fig.suptitle(full_title)
    ax.set(xlabel = label_0, ylabel = label_1)

    ax.fill_between(
        X, Y_Q1, Y_Q3, alpha=0.4, facecolor="xkcd:bright green",
        label = "$\mathregular{Q_1-Q_3}$"
    )
    ax.plot(X, Y_Q2,"r",linewidth=2, label = "Median")

    if is_validation:
        ax.plot(x, y,"bo",markersize = 3, label = "Observed")


    ax.grid()
    plt.legend(reverse=True)

    return fig


# ================================================================================================
def plot_QuantileRegressionDepth_1D(depth, sample, distribution, title, units, width, height):

    # Estimate a conditional quantile
    if isinstance(sample, (list, tuple)):
        pvar_name = sample[1]
        sample = sample[0]
        var_names = tuple(sample.getDescription())
        var_names = var_names + (pvar_name,)
    else:
        var_names = tuple(sample.getDescription())

    if sample.getDimension()==1:
        x = sample[:, 0]
        is_validation = False
    elif sample.getDimension()==2:
        x = sample[:, 0]
        y = sample[:, 1]
        is_validation = True

    if is_validation:
        y0 = np.array(y)
        ym = [distribution.computeConditionalQuantile(0.5, xi) for xi in x]
        ym = np.reshape(ym, y0.shape)

        RMSE = np.sqrt(np.square(np.subtract(y0, ym)).mean())

    depth_name = depth.getDescription()[0]

    label_0 = get_label(depth_name, units)
    label_1 = get_label(var_names[1], units)

    ind = range(x.getSize())
    X = np.array(x).flatten()

    Y_Q1 = [distribution.computeConditionalQuantile(0.25, x[i]) for i in ind]
    Y_Q2 = [distribution.computeConditionalQuantile(0.5, x[i]) for i in ind]
    Y_Q3 = [distribution.computeConditionalQuantile(0.75, x[i]) for i in ind]

    H = np.array(depth).flatten().tolist()
    c_Q2 = ot.Curve(H, Y_Q2)
    c_Q2.setColor("red")
    c_Q2.setLineWidth(2)

    if is_validation:
        full_title = "Conditional Quantile Regression with " + \
                        title + \
                        " of {1} | {0}\nRMSE = {2:.3g}".format(*var_names, RMSE)
    else:
        full_title = "Conditional Quantile Regression with " + \
                        title + \
                        " of {1} | {0}".format(*var_names)

    fig, ax = plt.subplots(1, 1)
    dpi = fig.get_dpi()
    new_size = (width / dpi, height / dpi)
    fig.set_size_inches(new_size)
    fig.suptitle(full_title)
    ax.set(xlabel = label_0, ylabel = label_1)

    ax.fill_between(
        H, Y_Q1, Y_Q3, alpha=0.4, facecolor="xkcd:bright green",
        label = "$\mathregular{Q_1-Q_3}$"
    )

    ax.plot(H, Y_Q2,"r",linewidth=2, label = "Median")

    if is_validation:
        ax.plot(depth, y, "bo", markersize = 3, label = "Observed")


    ax.grid()
    plt.legend(reverse=True)

    data = ot.Sample(np.array(Y_Q2, ndmin=2).T)
    if is_validation:
        data.setDescription([y.getDescription()[0] + "_val"])
    else:
        data.setDescription([pvar_name + "_pred"])
    return fig, data


# ================================================================================================
def check_mesh_is_structured(coord, v):
    x = np.array(coord[:,0])
    y = np.array(coord[:,1])
    v = np.array(v)
    nv = v.shape[1]
    X, Ix = np.unique(x,return_inverse=True)
    Y, Iy = np.unique(y,return_inverse=True)
    if X.size * Y.size == x.size:
        V = np.ndarray((nv,Y.size,X.size))
        for k in range(nv):
            for i in range(x.size):
                V[k,Iy[i],Ix[i]] = v[i,k]
        V = V.squeeze()
        return True, X, Y, V, Ix, Iy
    else:
        return False, x, y, v, None, None

# ================================================================================================
def plot_QuantileRegressionDepth_2D(coord, sample, distribution, title, units, width, height, interpolate=True, colmap = 'rainbow'):

    if isinstance(sample, (list, tuple)):
        pvar_name = sample[1]
        sample = sample[0]
        var_names = tuple(sample.getDescription())
        var_names = var_names + (pvar_name,)
    else:
        var_names = tuple(sample.getDescription())

    coord_names = tuple(coord.getDescription())

    if sample.getDimension()==1:
        x = sample[:, 0]
        is_validation = False
    elif sample.getDimension()==2:
        x = sample[:, 0]
        y = sample[:, 1]
        is_validation = True

    ind = range(x.getSize())

    y_Q1 = np.array([distribution.computeConditionalQuantile(0.25, x[i]) for i in ind],ndmin=2).T
    y_Q2 = np.array([distribution.computeConditionalQuantile(0.5, x[i]) for i in ind],ndmin=2).T
    y_Q3 = np.array([distribution.computeConditionalQuantile(0.75, x[i]) for i in ind],ndmin=2).T

    y_P = np.concatenate((y_Q2, y_Q3-y_Q1), axis=1)
    F = ot.Sample(y_P)
    if is_validation:
        F.stack(y)
        f_min = np.min(F[:,[0,2]])
        f_max = np.max(F[:,[0,2]])
    else:
        f_min = np.min(F[:,0])
        f_max = np.max(F[:,0])
    is_str, xc, yc, FF, __, __ = check_mesh_is_structured(coord, F)

    cmap = mpl.colormaps[colmap]
    #f_levels = np.linspace(f_min,f_max, cmap.N)
    loc = ticker.MaxNLocator(cmap.N)
    f_levels = loc.tick_values(f_min, f_max)

    fig = plt.figure(layout='constrained')
    dpi = fig.get_dpi()

    title_fsize = 14
    stitle_fsize = 10
    axis_fsize = 8

    fig.suptitle("Conditional Quantile Regression with " + title,fontsize=title_fsize)

    fig.set_size_inches((width/dpi, height/dpi))

    label_v0 = get_label(var_names[1], units,suff=' (Observed)')
    label_v1 = get_label(var_names[1], units,suff=' (Predicted)')
    label_iqr = get_label(var_names[1], units,suff=' (IQR)')

    label_c0 = get_label(coord_names[0], units)
    label_c1 = get_label(coord_names[1], units)

    if is_validation:
        axs = fig.subplots(3,1, sharex=True)
        axs[0].xaxis.set_tick_params(which='both', labeltop=True)
        axs[1].xaxis.set_tick_params(which='both', labeltop=True)
    else:
        axs = fig.subplots(2,1, sharex=True)
        axs[0].xaxis.set_tick_params(which='both', labeltop=True)

    if is_validation:
        ax = axs[0]
        ax.invert_yaxis()
        if interpolate:
            c = ax.contourf(xc, yc, FF[2,:,:], cmap=colmap, levels=f_levels)
        else:
            c = ax.pcolormesh(xc, yc, FF[2,:,:], cmap=colmap,shading='nearest', vmin=f_min, vmax=f_max)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(labelsize = axis_fsize)
        ax.set_xlabel(label_c0,fontsize=axis_fsize)
        ax.set_ylabel(label_c1,fontsize=axis_fsize)
        #ax.set_title(label_v0, y=-0.1, verticalalignment="top")
        ax.set_title(label_v0,fontsize=stitle_fsize)
        cbar = fig.colorbar(c, ax=ax, shrink=1.0, aspect=10)
        #cbar = colorbar(c)
        cbar.ax.tick_params(labelsize=axis_fsize)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_xlabel(units.get(var_names[1],''),fontsize=axis_fsize)

    if is_validation:
        ax = axs[1]
    else:
        ax = axs[0]
    ax.invert_yaxis()
    if interpolate:
        c = ax.contourf(xc, yc, FF[0,:,:], cmap=colmap, levels=f_levels)
    else:
        c = ax.pcolormesh(xc, yc, FF[0,:,:], cmap=colmap,shading='nearest', vmin=f_min, vmax=f_max)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize = axis_fsize)
    ax.set_xlabel(label_c0,fontsize=axis_fsize)
    ax.set_ylabel(label_c1,fontsize=axis_fsize)
    #ax.set_title(label_v1, y=-0.1, verticalalignment="top")
    ax.set_title(label_v1,fontsize=stitle_fsize)
    cbar = fig.colorbar(c, ax=ax, shrink=1.0, aspect=10)
    #cbar = colorbar(c)
    cbar.ax.tick_params(labelsize=axis_fsize)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel(units.get(var_names[1],''),fontsize=axis_fsize)

    iqr_min = np.min(F[:,1])
    iqr_max = np.max(F[:,1])
    cmap = mpl.colormaps[colmap]
    #iqr_levels = np.linspace(iqr_min,iqr_max, cmap.N)
    iqr_levels = loc.tick_values(iqr_min, iqr_max)

    if is_validation:
        ax = axs[2]
    else:
        ax = axs[1]
    ax.invert_yaxis()
    if interpolate:
        c = ax.contourf(xc, yc, FF[1,:,:], cmap=colmap, levels=iqr_levels)
    else:
        c = ax.pcolormesh(xc, yc, FF[1,:,:], cmap=colmap,shading='nearest', vmin=iqr_min, vmax=iqr_max)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize = axis_fsize)
    ax.set_xlabel(label_c0,fontsize=axis_fsize)
    ax.set_ylabel(label_c1,fontsize=axis_fsize)
    #ax.set_title(label_v1, y=-0.1, verticalalignment="top")
    ax.set_title(label_iqr,fontsize=stitle_fsize)
    cbar = fig.colorbar(c, ax=ax, shrink=1.0, aspect=10)
    #cbar = colorbar(c)
    cbar.ax.tick_params(labelsize=axis_fsize)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel(units.get(var_names[1],''),fontsize=axis_fsize)


    # ----------- separate figures --------------


    if is_validation:
        fig_A = plt.figure(layout='constrained')
        dpi = fig.get_dpi()
        fig_A.suptitle("Conditional Quantile Regression with " + title,fontsize=title_fsize)
        fig_A.set_size_inches((width/dpi, height/dpi))
        ax = fig_A.subplots()
        ax.invert_yaxis()
        if interpolate:
            c = ax.contourf(xc, yc, FF[2,:,:], cmap=colmap, levels=f_levels)
        else:
            c = ax.pcolormesh(xc, yc, FF[2,:,:], cmap=colmap,shading='nearest', vmin=f_min, vmax=f_max)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(labelsize = axis_fsize)
        ax.set_xlabel(label_c0,fontsize=axis_fsize)
        ax.set_ylabel(label_c1,fontsize=axis_fsize)
        ax.set_xlabel(label_c0)
        ax.set_ylabel(label_c1)
        #ax.set_title(label_v0, y=-0.1, verticalalignment="top")
        ax.set_title(label_v0,fontsize=stitle_fsize)
        cbar = fig.colorbar(c, ax=ax, shrink=1.0, aspect=10)
        #cbar = colorbar(c)
        cbar.ax.tick_params(labelsize=axis_fsize)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_xlabel(units.get(var_names[1],''),fontsize=axis_fsize)
    else:
        fig_A = None

    fig_B = plt.figure(layout='constrained')
    dpi = fig_B.get_dpi()
    fig_B.suptitle("Conditional Quantile Regression with " + title,fontsize=title_fsize)
    fig_B.set_size_inches((width/dpi, height/dpi))
    ax = fig_B.subplots()
    ax.invert_yaxis()
    if interpolate:
        c = ax.contourf(xc, yc, FF[0,:,:], cmap=colmap, levels=f_levels)
    else:
        c = ax.pcolormesh(xc, yc, FF[0,:,:], cmap=colmap,shading='nearest', vmin=f_min, vmax=f_max)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize = axis_fsize)
    ax.set_xlabel(label_c0,fontsize=axis_fsize)
    ax.set_ylabel(label_c1,fontsize=axis_fsize)
    ax.set_xlabel(label_c0)
    ax.set_ylabel(label_c1)
    #ax.set_title(label_v1, y=-0.1, verticalalignment="top")
    ax.set_title(label_v1,fontsize=stitle_fsize)
    cbar = fig.colorbar(c, ax=ax, shrink=1.0, aspect=10)
    #cbar = colorbar(c)
    cbar.ax.tick_params(labelsize=axis_fsize)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel(units.get(var_names[1],''),fontsize=axis_fsize)

    iqr_min = np.min(F[:,1])
    iqr_max = np.max(F[:,1])
    cmap = mpl.colormaps[colmap]
    #iqr_levels = np.linspace(iqr_min,iqr_max, cmap.N)
    iqr_levels = loc.tick_values(iqr_min, iqr_max)


    fig_C = plt.figure(layout='constrained')
    dpi = fig_C.get_dpi()
    fig_C.suptitle("Conditional Quantile Regression with " + title,fontsize=title_fsize)
    fig_C.set_size_inches((width/dpi, height/dpi))
    ax = fig_C.subplots()
    ax.invert_yaxis()
    if interpolate:
        c = ax.contourf(xc, yc, FF[1,:,:], cmap=colmap, levels=iqr_levels)
    else:
        c = ax.pcolormesh(xc, yc, FF[1,:,:], cmap=colmap,shading='nearest', vmin=iqr_min, vmax=iqr_max)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize = axis_fsize)
    ax.set_xlabel(label_c0,fontsize=axis_fsize)
    ax.set_ylabel(label_c1,fontsize=axis_fsize)
    ax.set_xlabel(label_c0)
    ax.set_ylabel(label_c1)
    #ax.set_title(label_v1, y=-0.1, verticalalignment="top")
    ax.set_title(label_iqr,fontsize=stitle_fsize)
    cbar = fig.colorbar(c, ax=ax, shrink=1.0, aspect=10)
    #cbar = colorbar(c)
    cbar.ax.tick_params(labelsize=axis_fsize)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel(units.get(var_names[1],''),fontsize=axis_fsize)
 # ----------- end separete figures --------------

    data = ot.Sample(np.array(y_Q2, ndmin=2))
    if is_validation:
        data.setDescription([y.getDescription()[0] + "_val"])
    else:
        data.setDescription([pvar_name + "_pred"])
    return fig, fig_A, fig_B, fig_C, data


# ================================================================================================
def calc_QuantileRegression(distribution, x, var_name=''):
    # ------------------------------------------------------------------------------------------------
    # ------ Quantile Regression------------------------------------------
    # Input:
    #   distribution - distribution 2D, result of QuantileRegression()
    #   x - 1 var, ot.Sample
    #   var_name - name of result variable, string
    # Output:
    #   regdata - result of regression, ot.Sample, 3 var - Median, Q1, Q3

    N = x.getSize()
    ind = range(N)
    y_Q1 = [distribution.computeConditionalQuantile(0.25, x[i]) for i in ind]
    y = [distribution.computeConditionalQuantile(0.5, x[i]) for i in ind]
    y_Q3 = [distribution.computeConditionalQuantile(0.75, x[i]) for i in ind]
    data = np.vstack((y,y_Q1,y_Q3)).T
    regdata = ot.Sample(data)
    if var_name:
        regdata.setDescription([var_name+' reg',var_name+' reg Q1',var_name+' reg Q3'])
    return regdata


# ================================================================================================
def QuantileRegression(coord, sample, distribution, title, units, width, height, interpolate=True, colmap = 'rainbow'):
    # ------ Quantile Regression------------------------------------------
    # Input:
    #   coord - ot.Sample, 1 var for 1D, 2 var for 2D
    #   sample:
    #       validation -> ot.Sample 2 var
    #       prediction -> tuple(ot.Sample 1 var, pred name: string)
    #   distribution - distribution
    #   title - title, string
    #   units - dictionary : variable name:variable units
    #   width, height - figure size
    #   interpolate:logical - True or False
    #   colmap - Colormap
    # Output:
    #   "Fig1" - View, if "prediction" -> None
    #   "Fig2" - View
    #   "Fig2a","Fig2b","Fig2b", in case 2D.  If "prediction" -> "Fig2A"==None
    #   "Data" - predicted or validated, ot.Sample, 1 var
    if isinstance(sample, (list, tuple)) and isinstance(sample[0], ot.Sample):
        is_validation = False
        if sample[0].getSize == 0:
            raise "Error in QuantileRegression: data is empty"
    elif isinstance(sample, ot.Sample):
        is_validation = True
        if sample.getSize == 0:
            raise "Error in QuantileRegression: data is empty"
    else:
        raise "Error in QuantileRegression: expected Sample or (Sample,string)"

    res = {}
    res["Fig1"] = plot_QuantileRegression(sample, distribution, title, units, width, height)
    if coord.getDimension() == 1:
        res["Fig2"], res["Data"] = plot_QuantileRegressionDepth_1D(
            coord, sample, distribution, title, units, width, height
        )
    elif coord.getDimension() == 2:
        res["Fig2"],res["Fig2a"],res["Fig2b"],res["Fig2c"], res["Data"] = plot_QuantileRegressionDepth_2D(
            coord, sample, distribution, title, units, width, height, interpolate=interpolate, colmap = colmap
        )

    return res


# ================================================================================================
def Scatterplot_OS(sample, simulation, sim_name, units, width, height):
    # Scatterplot------------------------------------------
    # Create a Pairs drawable
    # PointStyles() [bullet,circle,diamond,dot,fcircle,fdiamond,fsquare,ftriangleup,none,plus,square,star,times,triangledown,triangleup]

    var_names = tuple(sample.getDescription())
    label_v0 = get_label(var_names[0], units)
    label_v1 = get_label(var_names[1], units)

    fig, ax = plt.subplots(1, 1)
    dpi = fig.get_dpi()
    new_size = (width / dpi, height / dpi)
    fig.set_size_inches(new_size)
    fig.suptitle("Scatterplot")
    ax.set(xlabel = label_v0, ylabel = label_v1)

    ax.plot(sample[:,0],sample[:,1], "o", markeredgecolor = "blue", markerfacecolor = "cyan", label = "Data Sample", alpha = 0.5)
    ax.plot(simulation[:,0],simulation[:,1], "r.", label = "Simulation")
    plt.legend()
    plt.grid()

    return fig

# ================================================================================================
def plot_pdf(axis, copula):
    q = np.linspace(0.0, 1.0, 201)
    X, Y = np.meshgrid(q,q)
    x = X.flatten()
    y = Y.flatten()
    xy = np.stack((x, y), axis=-1)
    p = copula.computePDF(xy)
    P = np.array(p).reshape(X.shape)
    nlev = 21
    levels = np.zeros(nlev)
    for i in range(nlev):
        levels[i] = 0.25 * nlev / (nlev - i)
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.05, 1.05)
    CS = axis.contour(q,q,P,levels, colors = 'red')
    axis.clabel(CS, CS.levels, inline=True, fontsize=8)
    return

# ================================================================================================
def RankScatter(sample, marginals, copula, title, copula_name, units, width, height):
    var_names = tuple(sample.getDescription())
    # ------- Cloud in the rank space --------------
    ranksTransf = ot.MarginalTransformationEvaluation(
        marginals, ot.MarginalTransformationEvaluation.FROM
    )
    rankSample = ranksTransf(sample)
    rankCloud = ot.Cloud(rankSample, "blue", "fcircle", "sample")
    # ------- Graph with rank sample and estimated copula ---------------
    label_0 = get_label(var_names[0], units, "*")
    label_1 = get_label(var_names[1], units, "*")

##    graph = ot.Graph(title, label_0, label_1, True, "topleft")
##    graph.setLegendPosition("bottomleft")
##    # Then draw the iso-curves of the estimated copula
##
##    minPoint = [0.0] * 2
##    maxPoint = [1.0] * 2
##    pointNumber = [201] * 2
##    graphCop = copula.drawPDF(minPoint, maxPoint, pointNumber)
##    contour_estCop = graphCop.getDrawable(0)
##    # -------- Erase the labels of the iso-curves ----------
##    contour_estCop.setDrawLabels(False)
##    # Change the levels of the iso-curves
##    nlev = 21
##    levels = ot.Point(nlev)
##    for i in range(nlev):
##        levels[i] = 0.25 * nlev / (nlev - i)
##    contour_estCop.setLevels(levels)
##    # -------- Change the legend of the curves -------------
##    contour_estCop.setLegend("Parametric copula")
##    # ------- Change the color of the iso-curves -----------
##    contour_estCop.setColor("red")
##    contour_estCop.setDrawLabels(True)
##    # ------- Add the iso-curves graph into the cloud one -----------
##    graph.add(contour_estCop)
##    view = otv.View(graph, (width, height))
##    fig = view.getFigure()
##    ax = view.getAxes()[0]
##
##    ax.plot(rankSample[:,0],rankSample[:,1],'b.', label="Data sample")
##    handles, labels = ax.get_legend_handles_labels()
##    red_line = mlines.Line2D([], [], color="red")
##    handles.append(red_line)
##    labels.append(copula_name)
##    ax.legend(handles, labels)
##    graph.add(rankCloud)
##    rankCloud.setLegend("Data sample")
##
##    axes = view.getAxes()[0]
##    axes.get_children()[0].set_markersize(4)

    fig, ax = plt.subplots(1, 1)
    dpi = fig.get_dpi()
    new_size = (width / dpi, height / dpi)
    fig.set_size_inches(new_size)
    fig.suptitle(title)
    ax.set(xlabel = label_0, ylabel = label_1)
    plot_pdf(ax, copula)
    ax.plot(rankSample[:,0],rankSample[:,1],'b.', label="Data sample")

    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    line = mlines.Line2D([], [], color="red")
    handles.append(line)
    labels.append(copula_name)
    ax.legend(handles, labels)
    plt.grid()
    return fig


# ================================================================================================
def bivariate_par(
    sample,
    marginals,
    copulas,
    copula_name,
    units,
    width,
    height,
    par={"color1": "blue", "color2": "red"},
    ):
    # Case parametric
    # Input:
    #   sample_2v - sample with 2 variables
    #   marginals - list: marginal distribution[2]
    #   copulas - dictionary: key = name of copula, copulas[key] = copula factory
    #   copula_Name - string
    #       "BestBIC" - search best copula with BIC test
    #       "BestAIC" - search best copula with AIC test
    #       <name> - fixed copula name
    #   units - dictionary : variable name:variable units
    #   width, height - figure size
    #   par - colors
    # Output: dictionary
    #   "distribution" : best kernel smoothing distribution
    #   "CopulaName"   : Copula name
    #   "Copula"       : Copula
    #   test results:
    #   "test_aic" : test_aic
    #   "test_bic" : test_bic
    #   "test_RMSE" : RMSE
    #   "Simulation" : result of simulation
    #   plots:
    #   "Scatter"  : data scatter plot
    #   "SimScatter"  : view_Scatter
    #   "RankScatter" : view_RankScatter
    #   "PDF"      : view_PDF
    #   "CDF"      : view_CDF
    #   "RegrXY"   : Quantile Regression X vs Y
    #   "RegrDY"   : Quantile Regression Depth vs Y
    var_names = tuple(sample.getDescription())
    marginals[0].setDescription([var_names[0]])
    marginals[1].setDescription([var_names[1]])
    ##    res = {"distribution":None, "test_aic":None, "test_bic":None, "Simulation":None,
    ##            "Scatter":None, "RankScatter":None, "PDF":None, "CDF":None,
    ##            "RegrXY":None, "RegrDY":None, "Copula" : None}
    res = {}
    if copula_name == "BestCopulaModelBIC":
        copula_est, cname, _ = getBestCopula_BIC(copulas, marginals, sample)
        res["CopulaName"] = cname
        res["Copula"] = copula_est
        if not copula_est:
            return None
    elif copula_name == "BestCopulaModelAIC":
        copula_est, cname, _ = getBestCopula_AIC(copulas, marginals, sample)
        res["CopulaName"] = cname
        res["Copula"] = copula_est
        if not copula_est:
            return None
    else:
        try:
            copula_est = copulas[copula_name].build(sample)
            res["CopulaName"] = copula_name
            res["Copula"] = copula_est
        except:
            return None

    distribution_est = ot.ComposedDistribution(marginals, copula_est)
    res["distribution"] = distribution_est
    Np = distribution_est.getParameter().getDimension()
    res["test_aic"] = ot.FittingTest.AIC(sample, distribution_est, Np)
    res["test_bic"] = ot.FittingTest.BIC(sample, distribution_est, Np)
    res["test_RMSE"] = computeRMSE(sample, distribution_est)

    label_0 = get_label(var_names[0], units)
    label_1 = get_label(var_names[1], units)
    # Data Scatterplot------------------------------------------
    res["Scatter"] = plot_scatter_with_distrib(
        sample[:, 0],
        sample[:, 1],
        title="Scatter",
        namex=label_0,
        namey=label_1,
        size=(width, height),
    )

    # Simulation Scatterplot------------------------------------------
    # Create a Pairs drawable
    # PointStyles() [bullet,circle,diamond,dot,fcircle,fdiamond,fsquare,ftriangleup,none,plus,square,star,times,triangledown,triangleup]

    # Build joint distribution from marginal distributions and dependency structure

    # simulations from fitted parametric multivariate distribution_estution
    simulation = distribution_est.getSample(sample.getSize())
    simulation.setDescription(sample.getDescription())
    res["Simulation"] = simulation

    name = copula_est.getName()
    name = name.replace("Copula", " Copula")
    res["SimScatter"] = Scatterplot_OS(sample, simulation, name, units, width, height)


    # ------- Cloud in the rank space --------------
    res["RankScatter"] = RankScatter(
        sample,
        marginals,
        copula_est,
        "Parametric estimation of the copula",
        name,
        units,
        width,
        height,
    )
    # PDF------------------------------------------
    graph_pdf = ot.Graph("PDF", label_0, label_1, True, "topright")
    graph_pdf.add(distribution_est.drawPDF())
    view_PDF = otv.View(graph_pdf, (width, height))
    res["PDF"] = view_PDF.getFigure()

    # CDF------------------------------------------
    graph_cdf = ot.Graph("CDF", label_0, label_1, True, "topright")
    graph_cdf.add(distribution_est.drawCDF())
    view_CDF = otv.View(graph_cdf, (width, height))
    res["CDF"] = view_CDF.getFigure()

    return res


# ================================================================================================
def bivariate_ks(
    sample,
    marginals,
    kernels,
    kernel_name,
    units,
    width,
    height,
    par={"color1": "blue", "color2": "red"},
    ):
    # Case no parametric
    # Input:
    #   sample - sample with 2 variables
    #   marginals - list: marginal distribution[2]
    #   kernels - dictionary: key = name of kernel, kernels[key] = kernel distribution
    #   kernel_Name - string
    #       BestBIC - search best kernel with BIC test
    #       BestAIC - search best kernel with AIC test
    #       <name> - fixed kernel name
    # Output: dictionary
    #   "distribution" : best kernel smoothing distribution
    #   "KernelName"   : Kernel name
    #   test results:
    #   "test_aic" : test_aic
    #   "test_bic" : test_bic
    #   "test_RMSE" : RMSE
    #   "Simulation" : result of simulation
    #   plots:
    #   "Scatter"  : data scatter plot
    #   "SimScatter"  : view_Scatter
    #   "RankScatter" : view_RankScatter
    #   "PDF"      : view_PDF
    #   "CDF"      : view_CDF
    #   "RegrXY"   : Quantile Regression X vs Y
    #   "RegrDY"   : Quantile Regression Depth vs Y

    var_names = tuple(sample.getDescription())
    marginals[0].setDescription([var_names[0]])
    marginals[1].setDescription([var_names[1]])
    # Fit a non parametric copula using KernelSmoothing
    ##    res = {"distribution":None, "test_aic":None, "test_bic":None, "Simulation":None,
    ##            "Scatter":None, "RankScatter":None, "PDF":None, "CDF":None,
    ##            "RegrXY":None, "RegrDY":None, "Kernel":None}
    res = {}
    if kernel_name == "BestCopulaModelBIC":
        ks_copula, ks_kernel, kname, _ = getBestCopula_BIC_KS(
            kernels, marginals, sample
        )
        res["KernelName"] = kname
        if not ks_copula:
            return None
    elif kernel_name == "BestCopulaModelAIC":
        ks_copula, ks_kernel, kname, _ = getBestCopula_AIC_KS(
            kernels, marginals, sample
        )
        res["KernelName"] = kname
        if not ks_copula:
            return None
    else:
        try:
            ks_kernel = kernels[kernel_name]
            ks_copula = ot.KernelSmoothing(ks_kernel).build(sample).getCopula()
            res["KernelName"] = kernel_name
        except:
            return None
    ks_copula.setName("KS " + res["KernelName"])
    ks_distribution = ot.ComposedDistribution(marginals, ks_copula)
    res["distribution"] = ks_distribution
    # Np = ks_distribution.getParameter().getDimension()
    Np = 0
    res["test_aic"] = ot.FittingTest.AIC(sample, ks_distribution, Np)
    res["test_bic"] = ot.FittingTest.BIC(sample, ks_distribution, Np)
    res["test_RMSE"] = computeRMSE(sample, ks_distribution)

    # ------ non-conditional simulation ----------
    simulation_vector = ot.RandomVector(ks_distribution)
    simulation_size = sample.getSize()
    simulation = simulation_vector.getSample(simulation_size)
    simulation.setDescription(sample.getDescription())
    res["Simulation"] = simulation

    label_0 = get_label(var_names[0], units)
    label_1 = get_label(var_names[1], units)

    # Data Scatterplot------------------------------------------
    res["Scatter"] = plot_scatter_with_distrib(
        sample[:, 0],
        sample[:, 1],
        title="Scatter",
        namex=label_0,
        namey=label_1,
        size=(width, height),
    )

    # Simulation Scatterplot------------------------------------------
    # Create a Pairs drawable
    # PointStyles() [bullet,circle,diamond,dot,fcircle,fdiamond,fsquare,ftriangleup,none,plus,square,star,times,triangledown,triangleup]

    # Build joint distribution from marginal distributions and dependency structure

    # simulations from fitted non parametric multivariate distribution
    name = ks_kernel.getName()
    name = "KS(" + name + ")"
    res["SimScatter"] = Scatterplot_OS(sample, simulation, name, units, width, height)

    # ------- Cloud in the rank space --------------
    res["RankScatter"] = RankScatter(
        sample,
        marginals,
        ks_copula,
        "Non-parametric estimation of the copula",
        name,
        units,
        width,
        height,
    )

    # ------ Copula PDF------------------------------------------
    # graph_copula_pdf = ks_copula.drawPDF(minPoint, maxPoint, pointNumber)
    # view_copula_PDF = otv.View(graph_copula_pdf,(width,height))

    # ------ Copula CDF------------------------------------------
    # graph_copula_cdf = ks_copula.drawCDF(minPoint, maxPoint, pointNumber)
    # view_copula_CDF = otv.View(graph_copula_cdf,(width,height))

    # ------ Distribution PDF------------------------------------------
    graph_pdf = ot.Graph("PDF", label_0, label_1, True, "topright")
    graph_pdf.add(ks_distribution.drawPDF())
    view_distribution_PDF = otv.View(graph_pdf, (width, height))
    res["PDF"] = view_distribution_PDF.getFigure()

    # ------ Distribution CDF------------------------------------------
    graph_cdf = ot.Graph("CDF", label_0, label_1, True, "topright")
    graph_cdf.add(ks_distribution.drawCDF())
    view_distribution_CDF = otv.View(graph_cdf, (width, height))
    res["CDF"] = view_distribution_CDF.getFigure()

    return res


###from outliers import *
##from scipy import stats
##def find_distrib(sample,distributions):
##    best_bic = sys.float_info.max
##    best_distrib = None
##    for dname, distrib in distributions.items():
##        try:
##            d = distrib.build(sample)
##            bic = ot.FittingTest.BIC(sample, d)
##            print("{:15}: {:.2g}".format(dname, bic))
##            print(d)
##            if bic < best_bic:
##                best_distrib = d
##        except:
##            print("{:15}: ****".format(dname))
##    return best_distrib, best_bic
##
##from sklearn.neighbors import LocalOutlierFactor

# ================================================================================================
def get_description(model):
    res = {}
    res["var_names"] = list(model.getDescription())

    distr = model.getMarginal(0)
    res["marg1_is_parametric"] = distr.getImplementation().getName() != "KernelMixture"
    if res["marg1_is_parametric"]:
        # Parametric
        res["marg1_name"] = distr.getName()
        res["marg1_param_name"] = list(distr.getParameterDescription())
        res["marg1_param_value"] = list(distr.getParameter())
    else:
        # No Parametric
        res["marg1_kernel"] = distr.getImplementation().getKernel().getName()

    distr = model.getMarginal(1)
    res["marg2_is_parametric"] = distr.getImplementation().getName() != "KernelMixture"
    if res["marg2_is_parametric"]:
        # Parametric
        res["marg2_name"] = distr.getName()
        res["marg2_param_name"] = list(distr.getParameterDescription())
        res["marg2_param_value"] = list(distr.getParameter())
    else:
        # No Parametric
        res["marg2_kernel"] = distr.getImplementation().getKernel().getName()

    distr = model.getCopula()
    res["copula_is_parametric"] = distr.getParameterDimension() < 5
    if res["copula_is_parametric"]:
        # Parametric
        res["copula_name"] = distr.getName()
        res["copula_param_name"] = list(distr.getParameterDescription())
        res["copula_param_value"] = list(distr.getParameter())
    else:
        # No Parametric
        res["copula_kernel"] = distr.getImplementation().getName()
    return res
def print_res(title,res):
    print(title)
    for k, v in res.items():
        print(k," : ", type(v))
    return

def main():
    # TESTS FOR 2D
    Sample = ot.Sample.ImportFromCSVFile("test2_2d.csv", ",")
    Coord = Sample[:, 0:2]
    sample = Sample[:, [2,3]]
    units = {
        Sample.getDescription()[0]: "m",
        Sample.getDescription()[1]: "m",
        Sample.getDescription()[2]: "unit1",
        Sample.getDescription()[3]: "unit2",
        "VV":"unit3"
    }

    N = sample.getSize()
    distr = getUnivariateFactoriesList()
    marg1, _ = ot.FittingTest.BestModelBIC(sample[:, 0], list(distr.values()))
    marg2, _ = ot.FittingTest.BestModelBIC(sample[:, 1], list(distr.values()))
    marginals = (marg1, marg2)

    # TEST 1 - PARAMETRIC
    copulas = getCopulaFactoriesList()

    # best_copula_a, best_aic = getBestCopula_AIC(copulas, marginals, sample)
    # best_copula_b, best_bic = getBestCopula_BIC(copulas, marginals, sample)
    # res_1 = bivariante_par(sample,marginals,copulas,"Frank",600,400)
    res_1 = bivariate_par(sample, marginals, copulas, "BestCopulaModelBIC", units, 700, 600)
    print_res("2D par",res_1)
    plt.show()

    res = QuantileRegression(
        Coord, sample, res_1["distribution"], "Parametric copula (Validation)", units, 700, 600, interpolate=True,
        colmap='gist_rainbow_r',
    )
    plt.show()

    res = QuantileRegression(
        Coord, (sample[:,0],"VV"), res_1["distribution"], "Parametric copula (Prediction)", units, 700, 600, interpolate=False,
        colmap='gist_rainbow_r',
    )
    plt.show()

    units["depth"] = 'm'
    units["DEPTH"] = 'm'
    units["RHOB"] = 'g/sm^3'
    units["NPHI"] = 'vol'
    ## TESTS FOR 1D
    if 1:
        Sample = ot.Sample.ImportFromCSVFile("Z32.csv", ",")
        #        sample = Sample[:200,[10,14]]
        depth = Sample[:200, 0]

        sample = Sample[:200, [9, 12]]
        N = sample.getSize()
    ##    else:
    ##        mean = ot.Point((-1.0,2.0))
    ##        sigma = ot.Point((1.0,2.0))
    ##        R = ot.CorrelationMatrix(2, [1.0,-0.8, -0.8, 1.0])
    ##        test_distrib = ot.Normal(mean, sigma, R)
    ##        N = 500
    ##        sample = test_distrib.getSample(N)
    ##        depth = list(range(N))
    else:
        marginals = [ot.Normal(0.0, 0.5), ot.Normal(0.0, 0.9)]
        R = ot.CorrelationMatrix(2)
        R[0, 1] = -0.6
        copula = ot.NormalCopula(R)
        test_distrib = ot.ComposedDistribution(marginals, copula)
        N = 500
        sample = test_distrib.getSample(N)
        ##        h = np.array(range(N),ndmin=2)
        ##        v = np.sin(h/100)*2+2
        ##        w = np.square(v)+1
        ##        z = np.stack((v,w)).squeeze().T
        ##        sample = sample + z
        depth = ot.Sample(np.array(range(N), ndmin=2).T)

    distr = getUnivariateFactoriesList()
    marg1, _ = ot.FittingTest.BestModelBIC(sample[:, 0], list(distr.values()))
    marg2, _ = ot.FittingTest.BestModelBIC(sample[:, 1], list(distr.values()))
    marginals = (marg1, marg2)

    # TEST 1 - PARAMETRIC
    copulas = getCopulaFactoriesList()

    # best_copula_a, best_aic = getBestCopula_AIC(copulas, marginals, sample)
    # best_copula_b, best_bic = getBestCopula_BIC(copulas, marginals, sample)
    # res_1 = bivariante_par(sample,marginals,copulas,"Frank",600,400)
    res_1 = bivariate_par(sample, marginals, copulas, "BestCopulaModelBIC", units, 700, 600)
    plt.show()
    res = QuantileRegression(
        depth, sample, res_1["distribution"], "Parametric Copula", units, 700, 600
    )
    plt.show()

    res = QuantileRegression(
        depth, (sample[:,0],"TestV"), res_1["distribution"], "Parametric Copula", units, 700, 600
    )
    plt.show()

    # TEST 2 - NON PARAMETRIC
    kernel_distribution = ot.Epanechnikov()
    # Estimate Kernel Smoothing marginals
    kernel = ot.KernelSmoothing(kernel_distribution)
    est_var1 = kernel.build(sample[:, 0])
    est_var2 = kernel.build(sample[:, 1])
    marginals = [est_var1, est_var2]
    # best_copula_a, best_kernel_a, best_aic = getBestCopula_AIC_KS(kernels, marginals, sample)
    # best_copula_b, best_kernel_b, best_bic = getBestCopula_BIC_KS(kernels, marginals, sample)
    kernels = getKernels()
    res_2 = bivariate_ks(sample, marginals, kernels, "BestCopulaModelAIC", units, 700, 600)
    plt.show()
    res = QuantileRegression(
        depth, sample, res_2["distribution"], "No Parametric Copula", units, 700, 600
    )
    plt.show()

    res = QuantileRegression(
        depth, (sample[:,0],"TestV"), res_2["distribution"], "No Parametric Copula", units, 700, 600
    )
    plt.show()

if __name__ == "__main__":
    main()
