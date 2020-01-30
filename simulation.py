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
    #:  list: list of the data stored.
    input_data = []
    #:  list of str: files and directories in path.
    entries = os.listdir(path)
    #:  np array: list of redshift data and snapnums.
    redshift_data = np.loadtxt('redshifts.txt', delimiter=' ')
    redshift_data[redshift_data < 0.0] = 0.0

    for entry in entries:
        print('---Reading: ' + entry + '---', end='\r')
        #:  int: gets snapnum from filename.
        snap_num = int(entry.split('_')[3])
        #: flt: true redshift value of the file.
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
    #: dict of ndarrays: data columns with column title as the key.
    vals = {
        'fm' : input_snap.data[:,5],
        '$\delta$' : input_snap.data[:,4],
        '$\eta$' : input_snap.data[:,3],
        '|$\eta$-1|' : np.abs(input_snap.data[:,3]-1),
        'log(fm)' : np.log(input_snap.data[:,5]),
        'log($\delta$)' : np.log(input_snap.data[:,4]),
        'log($\eta$)' : np.log(input_snap.data[:,3]),
        'log(|$\eta$-1|)' : np.log(np.abs(input_snap.data[:,3]-1))
    }
    #: list of tuples: plot values. First is x, second is y.
    plots = [
        ('fm', '$\delta$'),
        ('$\eta$', 'fm'),
        ('$\eta$', '$\delta$'),
        ('|$\eta$-1|', 'fm'),
        ('|$\eta$-1|', '$\delta$'),
        ('log(fm)', 'log($\delta$)'),
        ('log($\eta$)', 'log(fm)'),
        ('log($\eta$)', 'log($\delta$)'),
        ('log(|$\eta$-1|)', 'log(fm)'),
        ('log(|$\eta$-1|)', 'log($\delta$)')
    ]
    for plot in plots:
        fig, ax = plt.subplots()
        x, y = vals[plot[0]], vals[plot[1]]
        ax.plot(x,y,'o',c='black',markersize=0.75)
        ax.plot(x,np.poly1d(np.polyfit(x, y, 1))(x),c='red',alpha=0.5)
        ax.set(
            xlabel = plot[0],
            ylabel = plot[1],
            title = 'Red Shift: ' + str(input_snap.r_shift)
        )
        plt.draw()
        fig.savefig('plots/'+str(input_snap.r_shift)+'_'+str(plot[0])+'_'+str(plot[1])+'.png')

def main():
    """Main function.

    Main body which runs methods.

    """
    #: list of np array: list of galaxy cluster data from input files.
    input_data = read_data('dat/')
    #: flt: test redshift value.
    target_redshift = .0
    for target in input_data:
        if target.r_shift == target_redshift:
            make_plots(target)

if __name__ == '__main__':
    main()
