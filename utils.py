"""Utilities Module

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import stats

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
                                               entry_data[:,1]>0.0,
                                               ~np.isnan(entry_data).any(axis=1)
                                               )]
        R200[rshift] = np.copy(entry_data[:, [0, 3, 4, 5]])
        R500[rshift] = np.copy(entry_data[:, [0, 8, 9, 10]])
    print()
    return R200, R500

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
    return y_calc, popt, pcov, pcoef

def plot_data(x, y, y_calc, xlab, ylab, sup_title, filename):
    """Method for making plots of input data.

    Uses matplotlib to plot a scatter of raw data and then a linear straight
    line of best fit on top. The plot is then saved to the plots/ directory.

    Args:
        x (ndarray): x values.
        y (ndarray): y values.
        y_calc (ndarray): Best fit y values.
        xlab (str): Label for the x axis.
        ylab (str): Label for the y axis.
        sup_title (str): Title for the plot.
        filename (str): Name that the plot will be stored under.

    """
    #: Fig, ax objects: New figure and axis created by matplotlib.
    fig, ax = plt.subplots()
    #: Plot of points.
    ax.plot(x,y,'o',c='black',markersize=0.75)
    ax.plot(x, y_calc,c='red',alpha=0.5,linewidth=0.5)
    ax.set(
        xlabel = xlab,
        ylabel = ylab,
        title = sup_title

    )
    plt.draw()
    fig.savefig('plots/'+filename+'.png')

def create_plots(vals, plots, rshift='NaN', log=False):
    """For making multiple plots of various data points.

    Recieves an input of tuples corresponding to data points to be plotted. Uses
    the fit_data method to collect the correct parameters for linear best fit
    plotting. Calls the plot_data method to plot the data.

    Args:
        vals (dict): Values from input text files.
        plots (list): Tuples of plots. First value is the x data, second value
            is the v data. Values correspond to keys in vals.
        target (ClusterSnap): Target object. Only used for getting the redshift
            at this point.
        log (bool): Whether this should be a log-log plot. Defaults to False.

    """
    if log == False:
        for plot in plots:
            #: ndarrays: x and y values to be plotted/compared.
            x, y = vals[plot[0]], vals[plot[1]]
            #: flt: Gradient, y intercept and uncertainties of fitted data.
            y_calc, popt, pcov, pcoef = fit_data(x, y)
            print('Gradient: {} +/- {}'.format(popt[0], pcov[0][0]))
            print('y-intercept: {} +/- {}'.format(popt[1], pcov[1][1]))
            print('Pearson Correlation: {}'.format(pcoef))
            #: Create a plot of the data.
            plot_data(
                x,
                y,
                y_calc,
                xlab=plot[0],
                ylab=plot[1],
                sup_title='Red Shift: {}'.format(rshift),
                filename='{}_{}_{}'.format(rshift, plot[0], plot[1])
            )
    elif log == True:
        for plot in plots:
            #: ndarrays: x and y values to be plotted/compared.
            x, y = np.log(vals[plot[0]]), np.log(vals[plot[1]])
            #: flt: Gradient, y intercept and uncertainties of fitted data.
            y_calc, popt, pcov, pcoef = fit_data(x, y)
            print('Gradient: {} +/- {}'.format(popt[0], pcov[0][0]))
            print('y-intercept: {} +/- {}'.format(popt[1], pcov[1][1]))
            print('Pearson Correlation: {}'.format(pcoef))
            #: Create a plot of the data.
            plot_data(
                x,
                y,
                y_calc,
                xlab='log({})'.format(plot[0]),
                ylab='log({})'.format(plot[1]),
                sup_title='Red Shift: {}'.format(rshift),
                filename='{}_log({})_log({})'.format(rshift, plot[0],plot[1])
            )

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
    #: tuples: Fit data from the fm and delta plot.
    _, popt, pcov,_ = fit_data(fm, delta)
    theta = (delta + popt[0] * fm) / 2.0
    theta_err = (pcov[0][0] / 2.0) * (fm + delta)
    return theta, theta_err

def calc_alpha():
    """Function that calculates an alpha value.

    """
    pass

def calc_relax(theta, theta_err, eta, alpha):
    """Function that calculates relaxation parameters.

    Relaxation parameter is calculated from the following formula:
        R = alpha * theta + |eta -1|
    The value of alpha is decided by the input.

    Args:
        theta (ndarray): Values of theta.
        theta_err (ndarray): Values of the uncertainty in theta values.
        eta (ndarray): Values of eta.
        alpha (flt): Arbitrary alpha variable defined by input.

    Returns:
        r (ndarray): Calculated relaxation parameters.
        r_err (ndarray): Uncertainty in relaxation parameter.

    """
    r = alpha * theta + np.abs(eta - 1)
    r_err = 0.0
    return r, r_err
