"""Script to create the relaxation parameter.

"""
from utils import *

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
                ('fm', '$\delta$')
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
                #: flt: Gradient of line of best fit of data.
                make_plot(x, y, plot[0], plot[1], title, filename)
            print('Best Correlation: ' + str(best_corr[0]) + ', ' + best_corr[1])

if __name__ == '__main__':
    main()
