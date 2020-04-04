"""Utilities Module

"""

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy import optimize
from scipy import stats
from scipy import interpolate
from scipy.signal import find_peaks

def read_data(path):
    """Method for reading in data files.
    
    Reads in files located in data folder. Determines redshift from the
    filename. Stores each cluster ID as a key with eta, fm, delta, and redshift
    data as values.
    
    Args:
        path (str): Folder containing data files.
        
    Returns:
        clusters (dict): Dict of ndarrays containing data for each cluster.
    
    """
    #: ndarray: Redshift data and snapnums.
    rshifts = np.loadtxt('redshifts.txt', delimiter=' ')
    #: Correct redshift data so that negative values become 0 redshift.
    rshifts[:,2][rshifts[:,2] < 0.0] = 0.0

    #: dict: Empty for storing cluster information.
    clusters = {}
    for entry in os.listdir(path):
        #: str: location of file.
        subpath = os.path.join(path, entry)
        if os.path.isdir(subpath):
            #: Skip if not a file.
            continue
        print('---Reading: {}---'.format(entry), end='\r')
        #: int: Get snapnum from filename.
        snapnum = int(entry.split('_')[3])
        #: flt: Get corresponding rshift from reference file.
        rshift = rshifts[np.where(rshifts[:,0] == snapnum)[0]][0][2]
        #: ndarray: Load data from the file.
        entry_data = np.loadtxt(path + entry)
        #: Select only the necessary columns (id, eta, fm, delta).
        entry_data = entry_data[:, [0, 3, 4, 5]]
        #: Remove invalid data.
        entry_data = entry_data[np.logical_and(
                                               ~np.isnan(entry_data).any(axis=1),
                                               np.all(entry_data >= 0, axis=1))]

        #: Iterate over the clusters in the file.
        for cluster in entry_data:
            if clusters.get(cluster[0]) == None:
                #: Make a new entry in the dictionary.
                clusters[cluster[0]] = []
            #: Add data to list.
            clusters[cluster[0]].append([cluster[1], cluster[2], cluster[3], rshift])
    print()
    for key, cluster in clusters.items():
        #: ndarray: Convert cluster to ndarray.
        cluster = np.array(cluster)
        #: Sort data by redshift.
        cluster = cluster[cluster[:,3].argsort()]
        clusters[key] = cluster

    #: ndarray: Read in mass history data
    masses = np.loadtxt('G3X-Central-Masses-for-M200.txt', delimiter=' ')

    #: Select range of redshifts.
    rshifts = rshifts[:,2][1:]
    for key, cluster in clusters.items():
        #: int: Locate the index of the corresponding cluster.
        idx = np.where(masses[:,0] == key)[0]
        #: ndarray: Select mass history of cluster.
        mass_hist = masses[idx][0][1:]
        #: ndarray: Create double column of mass history and redshift.
        mass_hist = np.column_stack((mass_hist, rshifts))
        #: list: Empty list for storing cluster mass histories.
        cluster_mass_hist = []
        for snap in cluster:
            #: flt: Redshift pointer.
            rshift = snap[3]
            #: int: find correct mass value for rshift.
            idx = np.where(mass_hist[:,1] == rshift)[0]
            #: Add to list
            cluster_mass_hist.append(mass_hist[idx][0][0])
        #: Convert to array.
        cluster_mass_hist = np.array(cluster_mass_hist)
        #: Add as column to existing array.
        cluster = np.column_stack((cluster, cluster_mass_hist))
        clusters[key] = cluster
    return(clusters, rshifts)

def pull_rshift_set(cluster_data, rshift):
    """Method for pulling all cluster data from specific rshift.
    
    Args:
        cluster_data (dict): All cluster data.
        rshift (flt): Redshift value to be grabbed.
    
    Returns:
        eta (ndarray): All eta values.
        delta (ndarray): All delta values.
        fm (ndarray): All fm values.
    
    """
    #: list: Store eta, delta, and fm values.
    eta, delta, fm = [], [], []
    # Iterate over the clusters.
    for _, cluster in cluster_data.items():
        #: ndarray: Select only correct rshift data.
        data = cluster[cluster[:,3] == rshift]
        if data.size != 0:
            #: Append eta, delta, and fm values.
            eta.append(data[:,0][0])
            delta.append(data[:,1][0])
            fm.append(data[:,2][0])
    #: Turn lists into arrays.
    eta = np.array(eta)
    delta = np.array(delta)
    fm = np.array(fm)
    return(eta, delta, fm)

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
    alpha, beta = 1.5783445063587698, 0.7318368559637508
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
        xnew (ndarray): Grid of new x values.
        ynew (ndarray): New interpolated y values.
        yder (ndarray): New derivative y values.
    
    """
    #: Spline representation of curve in 2D space.
    tck = interpolate.splrep(x, y, s=0)
    #: ndarray: Array of new x-values.
    xnew = np.linspace(np.amin(x), np.amax(x), 10000)
    #: ndarray: New interpolated y values.
    ynew = interpolate.splev(xnew, tck, der=0)
    #: ndarray: New y derviative values.
    yder = interpolate.splev(xnew, tck, der=1)
    return(xnew, ynew, yder)

def track_relaxation(cluster_data, cluster_idx):
    """Function for tracking the relaxation of a cluster.
    
    Args:
        cluster_data (dict): All cluster data.
        cluster_idx (int): ID of specific cluster.
            
    Returns:
        r (ndarray): Relaxation parameters for the cluster.
        r_err (ndarray): Uncertainty in relaxation.
        rshifts (ndarray): Corresponding redshift values.
    
    """
    cluster = cluster_data[cluster_idx]
    eta, delta, fm, rshift = cluster[:,0], cluster[:,1], cluster[:,2], cluster[:,3]

    # Calculate theta.
    theta, theta_err = calc_theta(fm, delta)
    # Calculate r.
    r, r_err = calc_r(theta, theta_err, eta)
    return(r, r_err, rshift)

def peak_limits(x, y, threshold, ax=None):
    """Function for getting the limits either side of peaks.
    
    Recieves a data set in the form of x and y and identifies the peaks of the data.
    Then iterates over the data either side of the peak and finds the first points
    where the data dips below the threshold value. Returns the two dip indexes and
    also plots the points on an axis if indicated.
    
    Args:
        x (ndarray): X values of the data.
        y (ndarray): Y values of the data.
        threshold (flt): Relaxation threshold value.
        ax (axis): Axis for data to be plotted on. Defaults to 'None'.
    
    Returns:
        dips (ndarray): Contains left and right dips of peaks.
        
    """
    #: ndarray: Index of peak values of data.
    peaks, _ = find_peaks(y, prominence=0.5)
    # Plot peaks.
    if ax != None:
        ax.plot(x[peaks], y[peaks], 'x')
        
    #: list: Store dip indexes.
    dips = [] 
    # Iterate over peaks.
    for peak in peaks:
        # Only use peaks found above the threshold.
        if y[peak] >= threshold:
            #: int: Counter for iteration.
            counter = 0
            #: int: Index position of left and right dip. Default 'None'
            left_dip, right_dip = None, None
            
            while left_dip == None or right_dip == None:
                try:
                    if y[peak + counter] <= threshold and right_dip == None:
                        right_dip = peak + counter
                except:
                    right_dip = 'NO LIMIT'
                try:
                    if y[peak - counter] <= threshold and left_dip == None:
                        left_dip = peak - counter
                except:
                    left_dip = 'NO LIMIT'
                counter += 1
            if left_dip != 'NO LIMIT' and right_dip != 'NO LIMIT':
                dips.append([left_dip, right_dip])
            if ax != None:
                if left_dip != 'NO LIMIT':
                    ax.plot(x[left_dip], y[left_dip], 'x', color='green')
                if right_dip != 'NO LIMIT':
                    ax.plot(x[right_dip], y[right_dip], 'x', color='green')
    #: ndarray: Transform dips into an ndarray for ease.
    dips = np.array(dips)
    return(dips)

def lookback_time(z):
    """Function for converting redshifts into lookback times.
    
    Uses: (2/(3Ho) * (1 - 1/(1+z)^(3/2))
    
    Args:
        z (flt): Redshift.
        
    Returns:
        t (flt): Timescale in Gyrs.
    
    """
    #: flt: Hubble constant.
    Ho = 72
    t = (2/(3*Ho)) * (1 - 1/((1+z)**(3/2)))
    return(t)