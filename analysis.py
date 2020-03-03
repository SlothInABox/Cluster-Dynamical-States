"""Script to create the relaxation parameter.

"""
from utils import *

def main():
    #: dicts: Dictionarys of the R200 and R500 data.
    R200, R500 = read_data('dat/')
    #: flt: Set the test alpha parameter.
    for rshift, params in R200.items():
        #: ndarrays: Parameters from values.
        eta, delta, fm = params[:,1], params[:,2], params[:,3]
        #: ndarrays: Theta values and uncertainties.
        theta, theta_err = calc_theta(fm, delta)
        if rshift == 0.0:
            fig, ax = plt.subplots()
            settings = dict(xlabel='R', ylabel='Frequency',
                            title='Distribution of Relaxation Parameters for very low \u03B1 values')
            ax.set(**settings)
            for alpha in np.linspace(0.0, 0.25, 5):
                r, r_err = calc_relax(theta, theta_err, eta, alpha)
                kde = stats.gaussian_kde(r)
                xgrid = np.linspace(0, np.amax(r), 1000)
                hist, bin_edges = np.histogram(r)
                kwargs = dict(label='\u03B1 = {}'.format(alpha),
                              markersize=0.75, )
                # line = ax.hist(r, **kwargs)
                line = ax.plot(xgrid, kde(xgrid), **kwargs)
                # ax.fill_between(xgrid, 0, kde(xgrid), alpha=0.3)
            ax.legend()
            plt.draw()
            plt.show()
            fig.savefig('plots/{}.png'.format('r_distribution_vvvlow'))

if __name__ == '__main__':
    main()
