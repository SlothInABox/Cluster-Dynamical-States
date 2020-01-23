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
        #: np array: array of locations of NaN values
        where_are_NaNs = np.isnan(entry_data.data)
        entry_data.data[where_are_NaNs] = 0.0
        entry_data.data[entry_data.data < 0.0] = 0.0
        input_data.append(entry_data)
    print()
    return input_data

def test_plot(test_data):
    """Method to test input data and get a plot.

    Uses test data (redshift value 1) and attempts to plot.

    Args:
        test_data (array): A numpy array of the data for a particular redshift.

    """
    x = test_data[:, 3]
    y = test_data[:, 5]
    plt.figure('TEST DATA')
    plt.scatter(x, y, c='black', s=0.5)
    plt.xlabel('R200: eta')
    plt.ylabel('R200: fm')
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
            test_plot(target.data)

if __name__ == '__main__':
    main()
