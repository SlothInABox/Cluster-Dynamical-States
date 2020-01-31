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

def make_plot(x, y, x_label, y_label, sup_title, filename):
    """Method for making plots of a data file.

    Creates a plot using two columns of input data. Plots a scatter of x and y
    points initially. Then uses the numpy.polyfits method to generate a line of
    best fit for the data. The figure is saved to the "plots/" directory.

    Args:
        x (ndarray): x values to be plotted.
        y (ndarray): y values to be plotted.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        sup_title (str): Title for the figure.
        filename (str): Filename to be stored. Do not include the file extension
            as it is added automatically.

    Returns:
        grad (flt): Gradient of the line of best fit.

    """
    #: Fig, ax objects: New figure and axis created by matplotlib.
    fig, ax = plt.subplots()
    #: Plot of points.
    ax.plot(x,y,'o',c='black',markersize=0.75)
    grad, _ = np.polyfit(x, y, 1)
    #: Best fit line.
    ax.plot(x,np.poly1d(np.polyfit(x, y, 1))(x),c='red',alpha=0.5, linewidth=0.5)
    #: Set x, y and title labels.
    ax.set(
        xlabel = x_label,
        ylabel = y_label,
        title = sup_title
    )
    plt.draw()
    fig.savefig('plots/'+filename+'.png')
    return grad

def main():
    """Main function.

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
            ]
            best_corr = (0.0, 'None')
            for plot in plots:
                #: ndarrays: x and y values to be plotted/compared.
                x, y = vals[plot[0]], vals[plot[1]]
                #: flt: Pearson correlation coefficient of the two data columns.
                corr_coef = np.corrcoef(x,y)[1,0]
                if 1.0 - np.abs(corr_coef) < 1.0 - np.abs(best_corr[0]):
                    best_corr = (corr_coef, plot[0] + ' : ' + plot[1])
                #: str: title of the graph.
                title = 'Red Shift: ' + str(target.r_shift) + ', Pearson r value: ' + str(corr_coef)
                #: str: filename for the plot.
                filename = str(target.r_shift) + '_' + plot[0] + '_' + plot[1]
                #: Create a plot for the data.
                make_plot(x, y, plot[0], plot[1], title, filename)
            print('Best Correlation: ' + str(best_corr[0]) + ', ' + best_corr[1])

if __name__ == '__main__':
    main()
