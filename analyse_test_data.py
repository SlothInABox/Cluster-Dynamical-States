"""Script to create the relaxation parameter.

"""
from utils import *

def create_plots(vals, target):
    """Creating plots of data points.

    """
    #: list of tuples: plot keys. First is x, second is y.
    plots = [
        ('fm', '$\delta$')
    ]
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

def create_theta(vals, target):
    """Creating the theta variable and plotting it.

    Creates the theta parameter from the fm and delta parameter. Makes linear
    and log-log plots of theta against eta and |eta-1|.

    Args:
        vals (dict): Data columns.
        target (ClusterSnap): Target data object, should be redshift 0 for
            testing.

    """
    #: ndarrays: x and y values to be combined.
    x, y = vals['fm'], vals['$\delta$']
    _, popt, pcov,_ = fit_data(x,y)
    theta, theta_error = combine_param_lin(x, y, popt[0], pcov[0][0])
    plots = [
        '$\eta$',
        '|$\eta$-1|'
    ]
    for plot in plots:
        print('---Plotting: theta vs {}'.format(plot))
        y_calc, popt, pcov, pcoef = fit_data(vals[plot], theta)
        print('Pearson Correlation: {}'.format(pcoef))
        print('Gradient: {} +/- {}'.format(popt[0], pcov[0][0]))
        plot_data(
            vals[plot],
            theta,
            y_calc,
            xlab=plot,
            ylab='$\Theta$',
            sup_title='Red Shift: {}'.format(target.r_shift),
            filename='{}_{}_{}'.format(target.r_shift,plot,'$\Theta$')
        )
        print('---Plotting: log(theta) vs log({})'.format(plot))
        y_calc, popt, pcov, pcoef = fit_data(np.log(vals[plot]), np.log(theta))
        print('Pearson Correlation: {}'.format(pcoef))
        print('Gradient: {} +/- {}'.format(popt[0], pcov[0][0]))
        plot_data(
            np.log(vals[plot]),
            np.log(theta),
            y_calc,
            xlab='log({})'.format(plot),
            ylab='log($\Theta$)',
            sup_title='Red Shift: {}'.format(target.r_shift),
            filename='{}_{}_{}'.format(target.r_shift,'log({})'.format(plot),'log($\Theta$)')
        )

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
            # create_plots(vals, target)
            create_theta(vals, target)


if __name__ == '__main__':
    main()
