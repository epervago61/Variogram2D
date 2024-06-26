# -------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      EPervago
#
# Created:     05/03/2024
# Copyright:   (c) EPervago 2024
# Licence:     <your licence>
# -------------------------------------------------------------------------------
"""
estimator_list
    String list identifying the semi-variance estimator to be used. Defaults to the Matheron estimator.

model_list
    String list identifying the theoretical variogram function to be used to describe the experimental variogram. Can be one of:

def get_model_parameters_list(model_name):
    Return list of all parameter's names for the model.


def Trend_Analyses_2D(sample, units, width,height,
        ntrend=0, removetrend=False, colormap="jet"):

  Trend Analysis 2D
    Input:
       sample - sample with 3 columns : X,Y, vals
       width,height - windows width and height
       ntrend - integer : 0,1,2. Default 0. Trend order.
       removetrend - boolean. Default False. If True, before all calculating trend will be removed.
    Output:
       results : dictionary:
        plots:
           "Trend" : Fig_Trend
        vals : nparray, detrended values


def make_variogram_2D(sample, units, width, height, Nb = 20, Na = 36, angles_tol = np.pi/8, bandwidth = 8,
                estimator='matheron', max_dist = None, return_counts = True):

    Parameters
    ----------
    sample - sample with 3 columns : X, Y, Value
    width,height - windows width and height
    Nb : class:`int`, optional
        number of bins to create. If None is given, will be determined
        by Sturges’ rule from the number of points.
        Default: 20
    Na : class:`int`, optional
        number of azimuts to create. Must be even.
        Default: 18

    angles_tol : class:`float`, optional
        the tolerance around the variogram angle to count a point as being
        within this direction from another point (the angular tolerance around
        the directional vector given by angles)
        Default: `np.pi/8` = 22.5°

    bandwidth : class:`float`, optional
        bandwidth to cut off the angular tolerance for directional variograms.
        If None is given, only the angles_tol parameter will control the point selection. Default: None

    estimator : :class:`str`, optional
        the estimator function, possible choices:

            * "matheron": the standard method of moments of Matheron
            * "cressie": an estimator more robust to outliers
        Default: "matheron"

    max_dist : class:`float`, optional
        Cut of length for the bins. If None is given, it will be set
        to one third of the box-diameter from the given points.
        Default: None

    mesh_type : :class:`str`, optional
        'structured' / 'unstructured', indicates whether the pos tuple
        describes the axis or the point coordinates.
        Default: `'unstructured'`



    Returns
    -------
    Res - dictionary
        "data" : Data struct:
            bin_centers : (Nb), :class:`numpy.ndarray`
                The bin centers.
            azimuts : (Na), :class:`numpy.ndarray`
                The azimuts.
            dir_vario : (Na, Nb), :class:`numpy.ndarray`
                The estimated variogram values at bin centers.
                Is stacked if multiple `directions` (d>1) are given.
            counts : (Na, Nb), :class:`numpy.ndarray`, optional
                The number of point pairs found for each bin.
        "MEshType": string : 'structured' | 'unstructured'
        "Statistics": statistic table
        Figures:
            "Trend" : trend analyses
            "VariogramEmp_2D" : experimental variograms 2D, isolines
            "VariogramEmp_1D" : experimental variogram on two direction

def fit_variogram_2D(
        data, units, width, height, param_list, model_name="Stable", param_init={}, param_bounds={}, NS=100,
        ):

    Parameters
    ----------
    data : Data struct: Created by  make_variogram_2D
        bin_centers : (Nb), :class:`numpy.ndarray`
            The bin centers.
        azimuts : (Na), :class:`numpy.ndarray`
            The azimuts.
        dir_vario : (Na, Nb), :class:`numpy.ndarray`
            The estimated variogram values at bin centers.
            Is stacked if multiple `directions` (d>1) are given.
        counts : (Na, Nb), :class:`numpy.ndarray`, optional
            The number of point pairs found for each bin.
    width,height - windows width and height

    model_name: class:`str`: model name, from model_list
    param_list: list of class:`str`
        List of parameters for optimization. Full list can be taken from get_model_parameters_list(model_name).
        All parameters missing from this list will be fixed.

    param_init: dictionary str:float, optional
        Dictionary of init values of parameters. Key is parameter name.

    param_bound: dictionary str:(float,float), optional
        Dictionary of bounds values of parameters. Key is parameter name.

    NS: class `int`, optional. Default: 100
        Number of point for cross-validation


    Returns
    -------
    Res - dictionary
        "model" - teoric model
        "vario_t" - teoric variogramm
        Figuras:
            "VariogramEmp_2D" : experimental and teoric variograms 2D, isolines
            "VariogramEmp_1D" : experimental and teoric variogram on two direction
            "CrossValidation" : crossvalidation
"""
import sys
import numpy as np
import gstools as gs
from matplotlib import pyplot as plt
import scipy
import pandas as pd
import openturns as ot
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from tabulate import tabulate
##import progress.bar as pb
from Variogram2D.ScatterD import plot_scatter_with_distrib
from time import perf_counter

# String identifying the semi-variance estimator to be used. Defaults to the Matheron estimator. Possible values are:
estimator_list = {
    "matheron": "Matheron",  # default
    "cressie": "Cressie-Hawkins",
}

# String identifying the theoretical variogram function to be used to describe the experimental variogram. Can be one of:
model_list = {'Gaussian':'Gaussian',
'Exponential':'Exponential',
'Matern':'Matern',
'Integral':'Integral',
'Stable':'Stable',
'Rational':'Rational quadratic',
'Cubic':'Cubic',
'Linear':'Bounded linear',
'Circular':'Circular',
'Spherical':'Spherical',
'HyperSpherical':'Hyper-Spherical',
'SuperSpherical':'Super-Spherical',
'JBessel':'J-Bessel',
'TPLGaussian':'Truncated-Power-Law with Gaussian',
'TPLExponential':'Truncated-Power-Law with Exponential',
'TPLStable':'Truncated-Power-Law with Stable',
'TPLSimple':'Simply truncated power law',
}

class var_data:
    def __init__(self):
        self.X = []
        self.Y = []
        self.F = []
        self.Azimuts = []
        self.Bins = []
        self.DirVario = []
        self.Model = None
        self.Counts = []
        self.MeshType = ''
        self.VarNames = []

# ================================================================================================
def get_label(var_name, units, suff=""):
    label = var_name + suff
    if var_name in units:
        label += " [" + units.get(var_name,'') + "]"
    return label
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


def get_model(model_name):
    return getattr(sys.modules["gstools"], model_name)(dim=2)

def get_model_list():
    return gs.covmodel.models.__all__+gs.covmodel.tpl_models.__all__

def get_model_par_names(model):
    return model.arg

def get_model_par_list(model_name):
    return get_model(model_name).arg

def set_model_param(model, name, value):
    setattr(model, name, value)

def get_model_param(model, name):
    value = getattr(model, name)
    if not isinstance(value, float):
        value = value[0]
    return value

def set_model_param_bounds(model, name, value):
    #if name != 'angles':
    setattr(model, name+'_bounds', value)

def get_model_param_bounds(model, name):
    if name in model.arg_bounds:
        return model.arg_bounds[name]
    else:
        return getattr(model, name+'_bounds')

def get_max_dist(X,Y):
    x1 = np.min(X)
    x2 = np.max(X)
    y1 = np.min(Y)
    y2 = np.max(Y)
    d = np.linalg.norm((x2-x1,y2-y1))
    return d/3

def get_num_bins(N_points):
    return int(np.ceil(2 * np.log2(N_points) + 1))


def set_model_param_bounds_default(model, name):
    if name == 'angles':
        model.angles_bounds = [0, np.pi,'co']
    elif name == 'anis':
        model.anis_bounds = [1.0, 100.0,'cc']

def get_model_param_value_list(model, param_list):
    res = []
    for pname in param_list:
        res.append(get_model_param(model, pname))
    return res

def set_model_param_value_list(model, param_list, values):
    for pname, value in zip(param_list,values):
        set_model_param(model, pname, value)

def get_model_param_bounds_list(model, param_list):
    ub = []
    lb = []
    for pname in param_list:
        b = get_model_param_bounds(model, pname)
        lb.append(b[0])
        ub.append(b[1])
    return lb,ub


def Trend_Analyses_2D(sample, units, width,height,
        ntrend=0, removetrend=False, colormap="jet"):

    # Trend Analysis 2D
    # Input:
    #   sample - sample with 3 columns : X,Y, vals
    #   width,height - windows width and height
    #   ntrend - integer : 0,1,2. Default 0. Trend order.
    #   removetrend - boolean. Default False. If True, before all calculating trend will be removed.
    # Output:
    #   results : dictionary:
    #    plots:
    #       "Trend" : Fig_Trend
    #    vals : nparray, detrended values

    results = {}
    N = sample.getSize()
    var_names = sample.getDescription()

    label_x = get_label(var_names[0], units)
    label_y = get_label(var_names[1], units)
    label_v = get_label(var_names[2], units)
    labels = [label_x, label_y, label_v]
    X = np.array(sample[:,0]).squeeze()
    Y = np.array(sample[:,1]).squeeze()
    V = np.array(sample[:,2]).squeeze()

    is_str, x, y, v, ix, iy = check_mesh_is_structured(sample[:,0:2], sample[:,2])

#    Fig_Trend = plt.figure(layout='constrained')
    Fig_Trend = plt.figure()
    dpi = Fig_Trend.get_dpi()
    new_size = (width / dpi, height / dpi)
    Fig_Trend.set_size_inches(new_size)

    if ntrend > 0:
        ncol = 3
    else:
        ncol = 1

    grid = ImageGrid(Fig_Trend, 111,
                nrows_ncols = (1,ncol),
                axes_pad = 0.6,
                cbar_location = "right",
                cbar_mode="each",
                cbar_size="5%",
                cbar_pad=0.1
        )


    ax = grid[0]
    ax.title.set_text('Observed')
    ax.set_aspect('equal')

    x = x.squeeze()
    y = y.squeeze()
    v = v.squeeze()
    if is_str:
        cp = ax.contourf(x, y, v, cmap = colormap)
    else:
        cp = ax.tricontourf(X, Y, V, cmap = colormap)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    cbar = plt.colorbar(cp, cax=grid.cbar_axes[0])


    if ntrend > 0:
        if ntrend == 1:
            W = np.stack((np.ones((N,)), X, Y),axis=-1)
            C = np.linalg.lstsq(W, V, rcond=None)[0]
            Tr = np.matmul(W,C)
        elif ntrend == 2:
            W = np.stack((np.ones((N,)), X, Y, X*Y, X**2,Y**2),axis=-1)
            C = np.linalg.lstsq(W, V, rcond=None)[0]
            Tr = np.matmul(W,C)
        else:
            raise Exception(f"Wrong order of trend: {ntrend}")
        #print(C)

        tr = np.empty_like(v)
        if is_str:
            for i in range(N):
                tr[iy[i],ix[i]] = Tr[i]

        ax = grid[1]
        ax.title.set_text('Trend')
        ax.set_aspect('equal')
        if is_str:
            cp = ax.contourf(x, y, tr, cmap = colormap)
        else:
            cp = ax.tricontourf(X, Y, Tr, cmap = colormap)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[1])

        V_Tr = V - Tr
        if is_str:
            v_tr = v - tr
        ax = grid[2]
        ax.title.set_text('Detrended')
        ax.set_aspect('equal')
        if is_str:
            cp = ax.contourf(x, y, v_tr, cmap = colormap)
        else:
            cp = ax.tricontourf(X, Y, V_Tr, cmap = colormap)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[2])


    results["Trend"] = Fig_Trend

    if is_str:
        results["MeshType"] = "structured"
    else:
        results["MeshType"] = "unstructured"

    if removetrend and ntrend > 0:
        results["Vals"] = V_Tr
    else:
        results["Vals"] = V

    results["X"] = X
    results["Y"] = Y
    if is_str:
        results["x"] = x
        results["y"] = y
        if removetrend and ntrend > 0:
            results["vals"] = v_tr
        else:
            results["vals"] = v
    else:
        results["x"] = None
        results["y"] = None
        results["vals"] = None

    return results

def find_extr_azimuts(dir_v):
    N, Nb = dir_v.shape
    M = N//2
    D = np.zeros(M)
    for k in range(M):
        D[k] = np.sum(dir_v[k,:] - dir_v[k+M,:])**2
    kmax = D.argmax()
    return kmax, kmax+M

def make_variogram_2D(sample, units, width, height, Nb = 20, Na = 36, angles_tol = np.pi/8, bandwidth = 8,
                estimator='matheron', max_dist = None, return_counts = True):

    """
    Parameters
    ----------
    sample - sample with 3 columns : X, Y, Value
    width,height - windows width and height
    Nb : class:`int`, optional
        number of bins to create. If None is given, will be determined
        by Sturges’ rule from the number of points.
        Default: 20
    Na : class:`int`, optional
        number of azimuts to create. Must be even.
        Default: 18

    angles_tol : class:`float`, optional
        the tolerance around the variogram angle to count a point as being
        within this direction from another point (the angular tolerance around
        the directional vector given by angles)
        Default: `np.pi/8` = 22.5°

    bandwidth : class:`float`, optional
        bandwidth to cut off the angular tolerance for directional variograms.
        If None is given, only the angles_tol parameter will control the point selection. Default: None

    estimator : :class:`str`, optional
        the estimator function, possible choices:

            * "matheron": the standard method of moments of Matheron
            * "cressie": an estimator more robust to outliers
        Default: "matheron"

    max_dist : class:`float`, optional
        Cut of length for the bins. If None is given, it will be set
        to one third of the box-diameter from the given points.
        Default: None

    mesh_type : :class:`str`, optional
        'structured' / 'unstructured', indicates whether the pos tuple
        describes the axis or the point coordinates.
        Default: `'unstructured'`



    Returns
    -------
    Res - dictionary
        "data" : Data struct:
            bin_centers : (Nb), :class:`numpy.ndarray`
                The bin centers.
            azimuts : (Na), :class:`numpy.ndarray`
                The azimuts.
            dir_vario : (Na, Nb), :class:`numpy.ndarray`
                The estimated variogram values at bin centers.
                Is stacked if multiple `directions` (d>1) are given.
            counts : (Na, Nb), :class:`numpy.ndarray`, optional
                The number of point pairs found for each bin.
        "MEshType": string : 'structured' | 'unstructured'
        "Statistics": statistic table
        Figures:
            "Trend" : trend analyses
            "VariogramEmp_2D" : experimental variograms 2D, isolines
            "VariogramEmp_1D" : experimental variogram on two direction
    """

    Azimuts = np.linspace(0,360,num=Na+1,endpoint=True)
    azimuts = np.deg2rad(Azimuts)

    var_names = tuple(sample.getDescription())
    res = Trend_Analyses_2D(sample, units, 1000,500,
        ntrend=2, removetrend=True, colormap="jet")

    results = {}
    results["Trend"] = res["Trend"]
    results["MeshType"] = res["MeshType"]
    mesh_type = results["MeshType"]
    X = res["X"]
    Y = res["Y"]
    Vals = res["Vals"]


    result = gs.vario_estimate(
        *((X, Y), Vals.T),
        angles = azimuts[:Na//2],
        angles_tol = angles_tol,
        bandwidth = bandwidth,
        mesh_type = "unstructured",
        return_counts = return_counts,
        #bin_edges = bins,
        bin_no = Nb,
        max_dist = max_dist,
        estimator = estimator,
    )
    if return_counts:
        bin_center, dir_v, counts = result
    else:
        bin_center, dir_v = result
    dir_vario = np.concatenate((dir_v,dir_v,dir_v[0:1,:]))

    # --------- Statistic table for 2 azimuts
    ia1, ia2 = find_extr_azimuts(dir_v)
    sa1 = "(" + str(Azimuts[ia1])+u"\u00b0" + ")"
    sa2 = "(" + str(Azimuts[ia2])+u"\u00b0" + ")"
    d = {
        "Lags": list(range(1, Nb + 1)),
        "Distance": bin_center,
        "Pairs "+sa1: counts[ia1,:],
        "Semivariances "+sa1: dir_v[ia1,:],
        "Pairs "+sa2: counts[ia2,:],
        "Semivariances "+sa2: dir_v[ia2,:],
    }
    stat_table = pd.DataFrame(d)
    results["Statistics"] = stat_table

    # Empirical variogram

    Fig_VariogramEmp_1D, ax = plt.subplots(1, 1)
    dpi = Fig_VariogramEmp_1D.get_dpi()
    new_size = (width / dpi, height / dpi)
    Fig_VariogramEmp_1D.set_size_inches(new_size)

    ax.plot(bin_center, dir_v[ia1,:], ".-", markersize=12, label="Empirical variogram "+sa1,
        color="cyan",
        markerfacecolor = "blue",
        alpha=0.8,
        lw=4,
    )
    for xl, yl, p in zip(bin_center, dir_v[ia1,:], counts[ia1,:]):
        label = "{:.0f}".format(p)
        plt.annotate(
            label,  # this is the text
            (xl, yl),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(10, 0),  # distance from text to points (x,y)
            color="b",
            ha="left",
            va="center",
        )  # horizontal alignment can be left, right or center

    ax.plot(bin_center, dir_v[ia2,:], ".-", markersize=12, label="Empirical variogram "+sa2,
        color="orange",
        markerfacecolor = "red",
        alpha=0.8,
        lw=4,
    )
    for xl, yl, p in zip(bin_center, dir_v[ia2,:], counts[ia2,:]):
        label = "{:.0f}".format(p)
        plt.annotate(
            label,  # this is the text
            (xl, yl),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(10, 0),  # distance from text to points (x,y)
            color="r",
            ha="left",
            va="center",
        )  # horizontal alignment can be left, right or center
    ax.axhline(Vals.var(), ls="--", color="g", lw=2, label="Variance")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Semivariance (%s)" % estimator_list[estimator])
    ax.grid(which="major")
    ax.legend(loc="best")
    results["VariogramEmp_1D"] = Fig_VariogramEmp_1D

    data = var_data()
    data.Azimuts = azimuts
    data.Bins = bin_center
    data.DirVario = dir_vario
    data.X = X
    data.Y = Y
    data.F = Vals
    data.MeshType = mesh_type
    data.bandwidth = bandwidth
    data.angles_tol = angles_tol
    data.estimator = estimator
    data.max_dist = max_dist
    data.return_counts = return_counts
    data.Nb = Nb
    data.VarNames = var_names
    results["data"] = data

    Fig_Empiric = plt.figure(layout='constrained')
    dpi = Fig_Empiric.get_dpi()
    new_size = (width / dpi, height / dpi)
    Fig_Empiric.set_size_inches((width/dpi, height/dpi))

    if return_counts:
        axs = Fig_Empiric.subplots(1,2,subplot_kw=dict(projection='polar'))
        ax = axs[0]
    else:
        ax = Fig_Empiric.subplots(1,1,subplot_kw=dict(projection='polar'))


    p1 = ax.contourf(azimuts, bin_center, dir_vario.T, 20, cmap = 'jet')
    ax.set_title("Empiric variogram")
    cbar = Fig_Empiric.colorbar(p1,fraction=0.1, pad=0.04, aspect=10)
    #cbar = fig.colorbar(p1,shrink=.6, pad=.05, aspect=10)

    if return_counts:
        counts = np.concatenate((counts,counts,counts[0:1,:]))
        data.counts = counts
        ax = axs[1]
        p2 = ax.contourf(azimuts, bin_center, counts.T, 20, cmap = 'jet')
        cbar = Fig_Empiric.colorbar(p2,fraction=0.1, pad=0.04, aspect=10)
        ax.set_title("Counts")

    results["VariogramEmp_2D"] = Fig_Empiric
    return results

def fit_variogram_2D(
        data, units, width, height, param_list, model_name="Stable", param_init={}, param_bounds={}, NS=100,
        ):
    """
    Parameters
    ----------
    data : Data struct: Created by  make_variogram_2D
        bin_centers : (Nb), :class:`numpy.ndarray`
            The bin centers.
        azimuts : (Na), :class:`numpy.ndarray`
            The azimuts.
        dir_vario : (Na, Nb), :class:`numpy.ndarray`
            The estimated variogram values at bin centers.
            Is stacked if multiple `directions` (d>1) are given.
        counts : (Na, Nb), :class:`numpy.ndarray`, optional
            The number of point pairs found for each bin.
    width,height - windows width and height

    model_name: class:`str`: model name, from model_list
    param_list: list of class:`str`
        List of parameters for optimization. Full list can be taken from get_model_parameters_list(model_name).
        All parameters missing from this list will be fixed.

    param_init: dictionary str:float, optional
        Dictionary of init values of parameters. Key is parameter name.

    param_bound: dictionary str:(float,float), optional
        Dictionary of bounds values of parameters. Key is parameter name.

    NS: class `int`, optional. Default: 100
        Number of point for cross-validation


    Returns
    -------
    Res - dictionary
        "model" - teoric model
        "vario_t" - teoric variogramm
        Figuras:
            "VariogramEmp_2D" : experimental and teoric variograms 2D, isolines
            "VariogramEmp_1D" : experimental and teoric variogram on two direction
            "CrossValidation" : crossvalidation
    """

    results = {}
    azimuts = data.Azimuts
    bin_center = data.Bins
    dir_vario = data.DirVario
    mesh_type = data.MeshType
    x = data.X
    y = data.Y
    field = data.F
    var_names = data.VarNames

    model = get_model(model_name)
    N = field.size

    def set_model_param_default(model, name):
        if name == "var":
            model.var = np.mean(dir_vario)
        elif name == "len_scale":
            model.len_scale = np.mean(bin_center)
        elif name == "angles":
            model.angles = 0.0
        elif name == "anis":
            model.anis = 2.0

    def calc_model_vario(coord, *args):
        set_model_param_value_list(model, param_list, args)
        vario = model.vario_spatial(coord)
        return vario

    # preproccessing
    R, AZ = np.meshgrid(bin_center, azimuts)
    X = R * np.cos(AZ)
    Y = R * np.sin(AZ)
    coord = np.vstack((X.reshape(1, -1), Y.reshape(1, -1)))
    vario = dir_vario.flatten()
    # set init values
    for pname, value in param_init.items():
        set_model_param(model, pname, value)
    # set default bounds
    for pname in param_list:
        set_model_param_bounds_default(model, pname)
    # set bounds
    for pname, value in param_bounds.items():
        set_model_param_bounds(model, pname, value)
    # print(model)
    # print(model.arg_bounds)
    p0 = get_model_param_value_list(model, param_list)
    lb, ub = get_model_param_bounds_list(model, param_list)
    # print('p0 = ',p0)
    # print('lb = ',lb)
    # print('ub = ',ub)
    # print(p0)
    popt, pcov = scipy.optimize.curve_fit(
        calc_model_vario, coord, vario, p0, bounds=(lb, ub), maxfev=1000000
    )
    # print(popt)
    set_model_param_value_list(model, param_list, popt)
    # print(model)
    vario_t = model.vario_spatial(coord)
    vario_t = vario_t.reshape(dir_vario.shape)
    results["model"] = model
    results["vario_t"] = vario_t

    #------------ Plot 2D variograms - empiric and teoric
    # fig, axs = plt.subplots(1,2,figsize=(10,5),subplot_kw=dict(projection='polar'))
    Fig_Variogram = plt.figure(layout='constrained')
    dpi = Fig_Variogram.get_dpi()
    new_size = (width / dpi, height / dpi)
    Fig_Variogram.set_size_inches(new_size)

    axs = Fig_Variogram.subplots(1,2, subplot_kw=dict(projection="polar"))
    Fig_Variogram.set_layout_engine("constrained")
    ax = axs[0]
    p1 = ax.contourf(azimuts, bin_center, dir_vario.T, 20, cmap = 'jet')
    ax.set_title("Empiric variogram")
    cbar = Fig_Variogram.colorbar(p1,fraction=0.1, pad=0.04, aspect=10)
    ax = axs[1]
    p2 = ax.contourf(azimuts, bin_center, vario_t.T, 20, cmap="jet")
    cbar = Fig_Variogram.colorbar(p2,fraction=0.1, pad=0.04, aspect=10)
    # -- Creating a new axes at the right side
    #ax3 = fig1.add_axes([0.9, 0.1, 0.03, 0.8])
    # -- Plotting the colormap in the created axes
    #cbar = fig1.colorbar(p2, cax=ax3)
    #fig1.subplots_adjust(left=0.05, right=0.85)
    # axs[0].set_title("Empiric variogram")
    ax.set_title("Fitting an anisotropic model: " + model.name)
    results["Variogram_2D"] = Fig_Variogram

    #------------ Plot variograms in two direction ----------------
    Fig_Variogram_1D = plt.figure(layout='constrained')
    dpi = Fig_Variogram_1D.get_dpi()
    new_size = (width / dpi, height / dpi)
    Fig_Variogram_1D.set_size_inches(new_size)

    ax = Fig_Variogram_1D.subplots()

    angle = model.angles[0]
    bin_center, dir_vario, counts = gs.vario_estimate(
        *((x, y), field.T),
        angles=[angle, angle + np.pi / 2],
        angles_tol=data.angles_tol,
        bandwidth=data.bandwidth,
        mesh_type="unstructured",
        return_counts=True,
        bin_no=data.Nb,
        max_dist=data.max_dist,
    )
    pl = ax.plot(
        bin_center,
        dir_vario[0],
        "b.",
        markersize=12,
        label=u"emp. vario on axis 0: {:.1f}\u00b0".format(np.rad2deg(angle)),
    )
    pl = ax.plot(
        bin_center,
        dir_vario[1],
        "r.",
        markersize=12,
        label=u"emp. vario on axis 1: {:.1f}\u00b0".format(np.rad2deg(angle) + 90),
    )

    for i, txt in enumerate(np.char.mod("%d", counts[0])):
        ax.annotate(
            txt,
            (bin_center[i] + 0.02, dir_vario[0, i]),
            xytext=(0, 4),  # 4 points vertical offset.
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="b",
        )
    for i, txt in enumerate(np.char.mod("%d", counts[1])):
        ax.annotate(
            txt,
            (bin_center[i] + 0.02, dir_vario[1, i]),
            xytext=(0, 4),  # 4 points vertical offset.
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="r",
        )
    x_s = np.linspace(0.0, bin_center[-1])
    axis=0
    ax.plot(x_s, model.vario_axis(x_s, axis),
        label = f"{model.name} variogram on axis {axis}",
        color="cyan",
        alpha=0.8,
        lw=4,
    )
    axis=1
    ax.plot(x_s, model.vario_axis(x_s, axis),
        label = f"{model.name} variogram on axis {axis}",
        color="orange",
        alpha=0.8,
        lw=4,
    )
    ax.axhline(data.F.var(), ls="--", color="g", lw=2, label="Variance")

    ax.legend(loc="lower right")
    ax.grid(which="major")

    ax.set_title("Fitting an anisotropic model: " + model.name)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Semivariance")
    results["Variogram_1D"] = Fig_Variogram_1D

    # Cross Validation (jacknife, leave one out)
    """
    Leave-one-out cross validation of the given variogram model using the
    OrdinaryKriging instance. This method can be called using Variogram.cross_validate.
    """
    # RMSE = V.cross_validate(method = 'jacknife', n = None, metric = 'rmse', seed=None)
    # print('----------------------------------------------------')
    # print('Variogram Cross Validation RMSE = ', RMSE)
    # print('----------------------------------------------------')
##    NS = 100  # Number of sample for cross validation calculation
##    NS = min(NS, N)
##    if valid_full or NS >= N:
##        index = range(N)
##    else:
##        index = random.sample(range(N), NS)
##    Z0 = field[index]
##    Z = np.full_like(Z0,np.inf)
##    bar = pb.Bar('Processing...', suffix='%(percent).1f%% - %(eta)ds',max=200)
##    for idx, k in zip(index, range(NS)):
##        f = field.copy()
##        f[idx] = np.inf
##        krig = gs.krige.Simple(model, (x,y),f, fit_variogram=True, exact=False)
##        Z[k] = krig((x[idx],y[idx]),return_var = False)
##        #print('.', end="")
##        #sys.stdout.flush()
##        bar.next()
##    #print()
##    bar.finish()
##    pN = 0.1
##    NS = round(N*pN)
    NS= 100
    NS = min(NS, N)
    index = random.sample(range(N), NS)
    Z0 = field[index]
    f = field.copy()
    f[index] = np.inf
#    t1 = perf_counter()
    krig = gs.krige.Simple(model, (x,y),f, fit_variogram=True, exact=False)
#    t2 = perf_counter()
#    print("Kriging time1: ", t2-t1)
#    Z = krig((x[index],y[index]),return_var = False, chunk_size=100)
    Z = krig((x[index],y[index]),return_var = False)
#    t3 = perf_counter()
#    print("Kriging time2: ", t3-t2)


    label_2 = get_label(var_names[2], units)
    label_2a = get_label(var_names[2], units, "*")
    Fig_CrossValidation = plot_scatter_with_distrib(
        Z0, Z, namex=label_2, namey=label_2a, size=(width, height)
    )
    axd = Fig_CrossValidation.get_axes()
    vmin = Z.min()
    vmax = Z.max()
    axd[3].axline((vmin, vmin), (vmax, vmax), linewidth=1, color="r")
    results["CrossValidation"] = Fig_CrossValidation
    return results



def main():
    M_name = 'Exponential'
    angle = np.deg2rad(40)
    model0 = get_model(M_name)
    angle = np.deg2rad(50)
    model0.angles = angle
    model0.len_scale = 10
    model0.anis = 2.0
    model0.var = 2.0
    print(model0)
    mesh_type = "unstructured"
    #mesh_type = "structured"
    if mesh_type == "structured":
        x = np.array(range(101))
        y = np.array(range(51))
        srf = gs.SRF(model0, seed=123456)
        field = srf((x, y), mesh_type="structured").T
        print(field.shape)

        fig = plt.figure()
        grid = ImageGrid(fig, 111,
                nrows_ncols = (1,3),
                axes_pad = 0.6,
                cbar_location = "right",
                cbar_mode="each",
                cbar_size="5%",
                cbar_pad=0.1
            )

        #fig, axs = plt.subplots(3,1,figsize=(4,12))
        ax = grid[0]
        ax.title.set_text('Modeled')
        ax.set_aspect('equal')
        cp = ax.contourf(x, y, field, cmap = 'jet')
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[0])


        X, Y = np.meshgrid(x,y)
        #field_tr = field + X*0.1 + Y*0.2 + 0.5*X*Y + 0.1*X**2 + 0.15*Y**2
        tr = X*0.05 + Y*0.1 - 0.005*X*Y + 0.001*(X-20)**2 +0.0005*(Y-40)**2
        ax = grid[1]
        ax.title.set_text('Trend')
        ax.set_aspect('equal')
        cp = ax.contourf(x, y, tr, cmap = 'jet')
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[1])

        ax = grid[2]
        ax.title.set_text('Trended')
        ax.set_aspect('equal')
        field_tr = field+tr
        cp = ax.contourf(x, y, field_tr, cmap = 'jet')
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[2])

    else:
        NP = 2000
        X = np.random.uniform(low=0.0, high=100.0, size=NP)
        Y = np.random.uniform(low=0.0, high=100.0, size=NP)
        srf = gs.SRF(model0, seed=123456)
        field = srf((X, Y), mesh_type="unstructured")

        fig = plt.figure()
        grid = ImageGrid(fig, 111,
                nrows_ncols = (1,3),
                axes_pad = 0.6,
                cbar_location = "right",
                cbar_mode="each",
                cbar_size="5%",
                cbar_pad=0.1
            )
        ax = grid[0]
        ax.title.set_text('Modeled')
        ax.set_aspect('equal')
        cp = ax.tricontourf(X, Y, field, cmap = 'jet')
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[0])

        tr = X*0.05 + Y*0.1 - 0.005*X*Y + 0.001*(X-20)**2 +0.0005*(Y-40)**2
        ax = grid[1]
        ax.title.set_text('Trend')
        ax.set_aspect('equal')
        cp = ax.tricontourf(X, Y, tr, cmap = 'jet')
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[1])

        ax = grid[2]
        ax.title.set_text('Trended')
        ax.set_aspect('equal')
        field_tr = field+tr
        cp = ax.tricontourf(X, Y, field_tr, cmap = 'jet')
        cbar = plt.colorbar(cp, cax=grid.cbar_axes[2])

    #fig.set_layout_engine("constrained")

    #ax.text(-0.5, 0.5, 'input test data', transform=ax.transAxes,
    #    fontsize=40, color='gray', alpha=0.5,
    #    ha='center', va='center', rotation=30)

    sample = ot.Sample(np.vstack((X.flatten().T,Y.flatten().T,field_tr.flatten().T)).T)

    sample.setDescription(['X','Y','F'])
    units = {'X':'m','Y':'m','F':'val'}
    sample.exportToCSVFile('test_2d_str.csv',',')
    #x, y, vals, results = Trend_Analyses_2D(sample, units, 2000,600,
    #    ntrend=2, removetrend=False, colormap="jet")
    mdist = 30
    nb = 20
    res = make_variogram_2D(sample,units,1000,700,Nb = nb, Na = 36, max_dist = mdist, bandwidth = 8, return_counts = True)
    print(res.keys())
    print(tabulate(res["Statistics"], headers='keys', tablefmt='psql',showindex=False))
    data = res["data"]
    # param_list = ['var', 'len_scale', 'anis', 'angles', 'alpha'] #Stable
    param_list = ["var", "len_scale", "anis", "angles"]  # Exponential
    res = fit_variogram_2D(data,units,1000,700, param_list, model_name=M_name, param_init={}, param_bounds={})
    print(res.keys())

    plt.show()

if __name__ == "__main__":
    main()
