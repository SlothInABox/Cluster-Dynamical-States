"""simulation file.

This module uses parameter data for assorted galaxy clusters to simulate
clusters at various timesteps. The data can be analysed to determine whether a
cluster is relaxed or unrelaxed. The timescale for a galaxy to go from relaxed
to unrelaxed can then be determined.

To do:
    *Create graphs of parameters for test cluster (zero redshift).

"""

import os
import numpy as np
import matplotlib.pyplot as plt

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
            data (ndarray): data inside input file.
            r_shift (flt): redshift value for the data file.

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
        print('---Reading: ' + entry + '---', end='\r')
        #:  int: gets snapnum from filename.
        snap_num = int(entry.split('_')[3])
        #: flt: true redshift value of the file.
        redshift = .0
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

def make_plots(input_snap, plots, vals):
    """Method for making plots of a data file.

    Creates a user inputted number of plots with input data columns. The data
    points are plottered on a scatter plot. A linear line of best fit is plotted
    using the numpy.polyfit method. The figures are saved to the 'plots/'
    directory.

    Args:
        input_snap (ClusterSnap): Target file object chosen for plotting.
        plots (list of tuples): User specified plots. Each tuple contains the
            key for the x-axis and y-axis.
        vals (dict): Specified input data columns. Each entry consists of a
            column name (key) and ndarray of data (value).

    """
    for plot in plots:
        #: Fig, ax objects: New figure and axis created by matplotlib.
        fig, ax = plt.subplots()
        #: ndarray: x and y values to be plotted.
        x, y = vals[plot[0]], vals[plot[1]]
        #: Plot of points.
        ax.plot(x,y,'o',c='black',markersize=0.75)
        #: Best fit line.
        ax.plot(x,np.poly1d(np.polyfit(x, y, 1))(x),c='red',alpha=0.5, linewidth=0.5)
        #: Set x, y and title labels.
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
            #: dict of ndarrays: data columns with column title as the key.
            vals = {
                'fm' : target.data[:,5],
                '$\delta$' : target.data[:,4],
                '$\eta$' : target.data[:,3],
                '|$\eta$-1|' : np.abs(target.data[:,3]-1),
                'log(fm)' : np.log(target.data[:,5]),
                'log($\delta$)' : np.log(target.data[:,4]),
                'log($\eta$)' : np.log(target.data[:,3]),
                'log(|$\eta$-1|)' : np.log(np.abs(target.data[:,3]-1)),
                'log(fm+$\delta$)' : np.log(target.data[:,4]+target.data[:,5])
            }
            #: list of tuples: plot keys. First is x, second is y.
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
                ('log(|$\eta$-1|)', 'log($\delta$)'),
                ('log(|$\eta$-1|)', 'log(fm+$\delta$)')
            ]
            make_plots(target, plots, vals)

if __name__ == '__main__':
    main()
