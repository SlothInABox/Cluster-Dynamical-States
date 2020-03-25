"""Utilities Module

"""

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy import optimize
from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import find_peaks

def read_data(path):
    """Function to read in data.

    Reads in files located in the 'dat' folder. Determines redshift from the
    filename and redshifts.txt file. Stores file data in dicitonarys of
    ndarrays.

    Any data rows which contain a NaN value or a negative value are incorrect
    and therefore removed from the data.

    R200 and R500 data is stored in separate dictionarys. Follows following
    format:
        [0]: Hid
        [1]: eta
        [2]: delta
        [3]: fm

    Args:
        path (str): The folder containing the data files.

    Returns:
        R200 (dict): R200 cluster data. Keys are the red shifts of the files.
            Data is an ndarray containing parameters.
        R500 (dict): R500 cluster data. Keys are the red shifts of the files.
            Data is an ndarray containing parameters.

    """
    #: list of str: Files and directories in defined path.
    entries = os.listdir(path)
    #:  np array: list of redshift data and snapnums.
    rshifts = np.loadtxt('redshifts.txt', delimiter=' ')
    #: Correct redshift data so that negative values become 0 redshift.
    rshifts[rshifts < 0.0] = 0.0
    #: dicts: Empty dictionarys for R200 and R500 data.
    R200, R500 = {}, {}
    for entry in entries:
        print('---Reading: {}---'.format(entry), end='\r')
        #: int: Get snapnum from filename.
        snap_num = int(entry.split('_')[3])
        #: flt: Cross referenced red shift value.
        rshift = rshifts[np.where(rshifts[:,0] == snap_num)][0][2]
        entry_data = np.loadtxt(path + entry)
        entry_data = entry_data[np.logical_and(
                                               ~np.isnan(entry_data).any(axis=1),
                                               np.all(entry_data >= 0, axis=1)
                                               )]
        R200[rshift] = np.copy(entry_data[:, [0, 3, 4, 5]])
        R500[rshift] = np.copy(entry_data[:, [0, 8, 9, 10]])
    print()
    return(R200, R500)

def fit_data(x, y):
    """Method for fitting data from a file.

    Fits data linearly using the scipy optimize.curve_fit method. Also finds the
    Pearson correlation coefficient using the scipy stats.pearsonr method.
    Calculates the best fit y values by passing least squares fit parameters
    into a straight line equation.

    Args:
        x (ndarray): x values to be plotted.
        y (ndarray): y values to be plotted.

    Returns:
        y_calc (ndarray): Best fitted data points.
        popt (list): Optimal values of parameters for minimizing the sum of the
            squared residuals.
        pcov (2d array): Estimated covariance of popt.
        pcoef (flt): Pearson correlation coefficient.

    """
    #: Linear best fit line.
    def line(x, m ,c):
        return m*x+c
    #: lists of flt: Least squares fit parameters and covariances.
    popt, pcov = optimize.curve_fit(line, x, y)
    #: flt: Pearson correlation coefficient from data.
    pcoef,_ = stats.pearsonr(x,y)
    #: ndarray: Best fitted line y values.
    y_calc = line(x, popt[0], popt[1])
    return(y_calc, popt, pcov, pcoef)

def calc_theta(fm, delta):
    """Creating the theta variable.

    Creates the theta parameter from the fm and delta parameter. Uses the
    gradient of the line of best fit between the two values to weight the two
    parameters.

    Args:
        fm (ndarray): Values of fm.
        delta (ndarray): Values of delta.

    Returns:
        theta (ndarray): New theta data.
        theta_err (ndarray): Uncertainty on each value of theta.

    """
    #: flts: Gradient weight factor and gradient weight uncertainty.
    m, m_err = 0.6032231653176229, 0.02965931345715287
#     _, popt, pcov,_ = fit_data(fm, delta)
#     m, m_err = popt[0], np.sqrt(pcov[0][0])
    #: ndarray: Values of theta using the gradient weight.
    theta = (delta + m * fm) / 2.0
    #: ndarray: Values of the uncertainty of theta.
    theta_err = (m_err / 2.0) * (fm + delta)
    return(theta, theta_err)

def get_distribution(data):
    """Method for getting the distribution of a data set.
    
    Performs a kernal density estimation on the data to smooth it.
    
    Args:
        data (ndarray): Data to be distributed.
    Returns:
        kde (kde object): Equation describing KDE of the data set.
        xgrid (ndarray): x-values passed into the KDE.
    
    """
    kde = stats.gaussian_kde(data)
    xgrid = np.linspace(0, np.amax(data), 1000)
    return(kde, xgrid)

def calc_r(theta, theta_err, eta):
    """Function for calculating the relaxation parameter.
    
    Uses alpha and beta values found through previous analysis.
    
    Calculates r using the following equation:
        r = alpha * theta + beta * |eta - 1|
    
    Args:
        theta (ndarray): Theta values.
        theta_err (ndarray): Uncertainties on theta values.
        eta (ndarray): Values of eta.
    
    Returns:
        r (ndarray): Relaxation parameter.
        r_err (ndarray): Uncertainty on relaxation parameter.
    
    """
    #: flts: Define alpha and beta.
    alpha, beta = 1.5988696706651493, 0.7037942890537529
    #: ndarray: Corrected eta values.
    abs_eta = np.abs(eta - 1)
    
    #: ndarray: Relaxation parameter.
    r = alpha * theta + beta * abs_eta
    #: ndarray: Uncertainty on relaxation parameter.
    r_err = np.sqrt((alpha**2) * (theta_err**2))
    return(r, r_err)

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colourings.
    
    Args:
        xs (ndarray): Container of x coordinates.
        ys (ndarray): Container of y coordinates.
        c (ndarray): Container of numbers mapped to colormap.
        ax (ax obj): Optional axis to plot on.
        kwargs (dict): Passed to LineCollection as settings.
        
    Returns:
        lc (obj): Line collection instance.
    
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    ax.add_collection(lc)
    ax.autoscale()
    return(lc)

def spline(x, y):
    """Make a curved line through data points.
    
    Args:
        x (ndarray): x values.
        y (ndarray): y values.
        
    Returns:
        xgrid (ndarray): Grid of new x values.
        power_smooth (ndarray): New smoothed y values.
    
    """
    xgrid = np.linspace(np.amin(x), np.amax(x), 10000)
    spl = make_interp_spline(x, y, k=3)
    power_smooth = spl(xgrid)
    return(xgrid, power_smooth)

def track_relaxation(cluster_idx, data):
    """Function for tracking the relaxation of a cluster.
    
    Args:
        cluster_idx (int): Index of the target cluster.
        data (dict): Contains parameter data. Keys are redshifts, values
            are ndarrays containing parameter data.
            
    Returns:
        r (ndarray): Relaxation parameters for the cluster.
        r_err (ndarray): Uncertainty in relaxation.
        rshifts (ndarray): Corresponding redshift values.
    
    """
    #: list: Stores relaxation values.
    r_cols = []
    
    for rshift in sorted(data.keys()):
        #: flts: Parameters for calculating theta and relaxation.
        eta, delta, fm = data[rshift][cluster_idx,1], data[rshift][cluster_idx,2], data[rshift][cluster_idx,3]
        
        #: flts: Calculated theta and uncertainty.
        theta, theta_err = calc_theta(fm, delta)
        #: flts: Calculated r and uncertainty.
        r, r_err = calc_r(theta, theta_err, eta)
        r_cols.append([r, r_err, rshift])
        
    #: ndarray: Convert r_cols into ndarray.
    r_cols = np.array(r_cols)
    #: ndarrays: Point to new parameters.
    r, r_err, rshift = r_cols[:,0], r_cols[:,1], r_cols[:,2]
    return(r, r_err, rshift)