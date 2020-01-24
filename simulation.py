"""simulation file.

This module uses parameter data for assorted galaxy clusters to simulate
clusters at various timesteps. The data can be analysed to determine whether a
cluster is relaxed or unrelaxed. The timescale for a galaxy to go from relaxed
to unrelaxed can then be determined.

To do:
    *Import data from 'dat' folder.
    *Create graphs of parameters for test cluster (zero redshift).

"""

import os
import numpy as np
import matplotlib.pyplot as plt

class ClusterSnap(object):
    """Class containing data of object file.

    This class contains the data for one input file as well as the corresponding
    redshift of the data in that file.

    """

    def __init__(self, data, r_shift):
        """Docstring on the __init__ method.

        Args:
            data (np array): data inside input file.
            r_shift (flt): redshift value for the data file.

        """
        super(ClusterSnap, self).__init__()
        self.data = data
        self.r_shift = r_shift


def read_data(path):
    """Function to read in data.

    Reads in files located in the 'dat' folder. Determines redshift from the
    filename and redshifts.txt file. Stores file data in lists.

    Args:
        path (str): The folder containing the data files.

    Returns:
        input_data (list): A list of input ClusterData objects.

    """
    #:  list: list of the data stored
    input_data = []
    #:  list of str: files and directories in path
    entries = os.listdir(path)
    #:  np array: list of redshift data and snapnums
    redshift_data = np.loadtxt('redshifts.txt', delimiter=' ')
    redshift_data[redshift_data < 0.0] = 0.0

    for entry in entries:
        print('---Reading: ' + entry + '---', end='\r')
        #:  int: gets snapnum from filename
        snap_num = int(entry.split('_')[3])
        #: flt: true redshift value of the file
        redshift = .0
        for row in redshift_data:
            if int(row[0]) == snap_num:
                redshift = row[2]
        #: ClusterSnap: object containing data inside file and redshift value.
        entry_data = ClusterSnap(np.loadtxt(path + entry, delimiter=' '), redshift)
        entry_data.data = entry_data.data[
            np.logical_and(
                entry_data.data[:,1]>0.0,
                ~np.isnan(entry_data.data).any(axis=1)
            )
        ]
        input_data.append(entry_data)
    print()
    return input_data

def make_plots(input_snap):
    """Method for making plots of a data file.

    Uses a ClusterSnap object to create plots of eta against delta and fm.
    Currently uses the R200 data columns.
    Straight line of best fit is plotted using the numpy.polyfit function.

    Args:
        input_snap (ClusterSnap): Object chosen for plotting.

    """
    #: figure, axis: generated figure and axis with matplotlib
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Red shift: ' + str(input_snap.r_shift))
    x = np.abs(input_snap.data[:,3] - 1)
    for i in range(0, 2):
        #: np arrays: polyfitted y values for creating straight line
        y, m = np.polynomial.polynomial.polyfit(x, input_snap.data[:,4+i], 1)
        axs[i].scatter(x, input_snap.data[:,4+i],c='black',s=0.5)
        axs[i].plot(x, y + m * x, c='red', alpha=0.5)
        axs[i].set(xlabel='R200: abs(eta-1)')
        if i == 0:
            axs[i].set(ylabel='R200: delta')
        elif i == 1:
            axs[i].set(ylabel='R200: fm')
    fig.savefig('plots/abs(eta-1)_'+ str(input_snap.r_shift) + '_plot.png')
    plt.show()

def main():
    """Main function.

    Main body which runs methods.

    """
    #: list of np array: list of galaxy cluster data from input files
    input_data = read_data('dat/')
    #: flt: test redshift value
    target_redshift = .0
    for target in input_data:
        if target.r_shift == target_redshift:
            make_plots(target)

if __name__ == '__main__':
    main()
