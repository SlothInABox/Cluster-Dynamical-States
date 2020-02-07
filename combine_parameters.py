"""Script to create the relaxation parameter.

"""
from utils import *

def create_theta(vals, target):
    """Creating the theta variable and plotting it.

    Creates the theta parameter from the fm and delta parameter. Makes linear
    and log-log plots of theta against eta and |eta-1|.

    Args:
        vals (dict): Data columns.
        target (ClusterSnap): Target data object, should be redshift 0 for
            testing.

    Returns:
        theta (ndarray): New theta data.
        theta_error (ndarray): Uncertainty on each value of theta.

    """
    #: ndarrays: x and y values to be combined.
    x, y = vals['fm'], vals['$\delta$']
    _, popt, pcov,_ = fit_data(x,y)
    theta, theta_error = combine_param_lin(x, y, popt[0], pcov[0][0])
    plots = [
        ('$\eta$', '$\Theta$'),
        ('|$\eta$-1|', '$\Theta$')
    ]
    vals['$\Theta$'] = theta
    create_plots(vals, plots, target,log=True)
    return theta, theta_error

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
            theta, theta_error = create_theta(vals, target)


if __name__ == '__main__':
    main()
