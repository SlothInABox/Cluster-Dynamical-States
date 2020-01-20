"""simulation file

This module uses parameter data for assorted galaxy clusters to simulate
clusters at various timesteps. The data can be analysed to determine whether a
cluster is relaxed or unrelaxed. The timescale for a galaxy to go from relaxed
to unrelaxed can then be determined.

To do:
    *Import data from 'dat' folder
    *Create graphs of parameters for test cluster (zero redshift)

"""

import os
import numpy as np

def read_data(path):
    """Function to read in data

    Reads in files located in the 'dat' folder. Determines redshift from the
    filename and redshifts.txt file. Stores file data in lists.

    Args:
        path (str): The folder containing the data files

    Returns:
        data (list): A list of the data. Each is it's own numpy array

    """
    #:  list: list of the data stored
    data = []
    #:  list of str: files and directories in path
    entries = os.listdir(path)
    #:  np array: list of redshift data and snapnums
    redshift_data = np.loadtxt('redshifts.txt', delimiter=' ')

    for entry in entries:
        print('---Reading: ' + entry + '---', end='\r')
        #:  int: gets snapnum from filename
        snap_num = int(entry.split('_')[3])
        #: flt: true redshift value of the file
        redshift = .0
        for row in redshift_data:
            if int(row[0]) == snap_num:
                redshift = row[2]
        #: np array: list of data inside .txt file
        entry_data = np.loadtxt(path + entry, delimiter=' ')
        #: list of int: shape of array for adding another column
        array_length = np.shape(entry_data)
        #: np array: numpy array of redshift value
        redshift_added = np.full((array_length[0], array_length[1]+1), redshift)
        redshift_added[:,:-1] = entry_data
        where_are_NaNs = np.isnan(redshift_added)
        redshift_added[where_are_NaNs] = 0.0
        redshift_added[redshift_added < 0.0] = 0.0
        data.append(redshift_added)
    print()
    return data

def test_plot(input_data):
    """Method to test input data and get a plot.

    Uses test data (redshift value 1) and attempts to plot

    Args:
        input_data (array): A numpy array of the data for a particular redshift

    """
    pass

def main():
    """Main function

    Main body which runs methods

    """
    data = read_data('dat/')
    #: flt: test redshift value
    target_redshift = .0

if __name__ == '__main__':
    main()
