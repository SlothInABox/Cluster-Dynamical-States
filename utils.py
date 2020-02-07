"""Utilities Module

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import stats

class ClusterSnap(object):
    """Class containing data of object file.

    This class contains the data for one input file as well as the corresponding
    redshift of the data in that file.

    Attributes:
        data (ndarray): Data from input file read in using numpy.
        r_shift (flt): Red shift value for particular data. Calculated from
            input file name.

    """
    def __init__(self, data, r_shift):
        """Docstring on the __init__ method.

        Args:
            data (ndarray): Data inside input file.
            r_shift (flt): Redshift value for the data file.

        """
        super(ClusterSnap, self).__init__()
        self.data = data
        self.r_shift = r_shift

def read_data(path):
    """Function to read in data.

    Reads in files located in the 'dat' folder. Determines redshift from the
    filename and redshifts.txt file. Stores file data in lists.
    Any data rows which contain a NaN value or a negative value are incorrect
    and therefore removed from the data.

    Args:
        path (str): The folder containing the data files.

    Returns:
        input_data (list): A list of input ClusterData objects.

    """
    #:  list: list of the data stored.
    input_data = []
    #:  list of str: files and directories in path.
    entries = os.listdir(path)
    #:  np array: list of redshift data and snapnums.
    redshift_data = np.loadtxt('redshifts.txt', delimiter=' ')
    redshift_data[redshift_data < 0.0] = 0.0
    for entry in entries:
        print('---Reading: {}---'.format(entry), end='\r')
        #:  int: gets snapnum from filename.
        snap_num = int(entry.split('_')[3])
        #: flt: true redshift value of the file.
        redshift = 0.0
        for row in redshift_data:
            if int(row[0]) == snap_num:
                redshift = row[2]
        #: ClusterSnap: object containing data inside file and redshift value.
        entry_data = ClusterSnap(
            np.loadtxt(path + entry, delimiter=' '),
            redshift
        )
        entry_data.data = entry_data.data[
            np.logical_and(
                entry_data.data[:,1]>0.0,
                ~np.isnan(entry_data.data).any(axis=1)
            )
        ]
        input_data.append(entry_data)
    print()
    return input_data

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

def create_plots(vals, plots, target, log=False):
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
                sup_title='Red Shift: {}'.format(target.r_shift),
                filename='{}_{}_{}'.format(target.r_shift, plot[0], plot[1])
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
                sup_title='Red Shift: {}'.format(target.r_shift),
                filename='{}_log({})_log({})'.format(target.r_shift, plot[0],
                                                     plot[1])
            )

def combine_param_lin(param1, param2, gradient, gradient_error):
    """Method for combining two parameters.

    Combines two parameters that have a linear relationship using the gradient
    of a best fit line between the two of them as a weight.

    Args:
        param1 (ndarray): First parameter.
        param2 (ndarray): Second parameter.
        gradient (flt): Gradient of best fit line for param2 (y) plotted against
            param1 (x).
        gradient_error (flt): Uncertainty on the gradient value.

    Returns:
        combined_param (ndarray): Combined parameter.
        combined_error (ndarray): Uncertainty on the combined parameter.

    """
    combined_param = (param2 + gradient*param1)/2.0
    combined_error = (gradient_error/2) * (param1 + param2)
    return combined_param, combined_error

def calc_relaxation(theta, eta):
    """Function that calculates relaxation parameter.

    Relaxation parameter, R = theta + alpha * eta. Loops through a number of
    alpha values to create R.

    """

    pass
