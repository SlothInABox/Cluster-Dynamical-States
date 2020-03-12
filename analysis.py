"""Script to create the relaxation parameter.

"""
from utils import *

def eta_distribution(eta):
    abs_eta = np.abs(eta - 1)

    fig, ax = plt.subplots()
    settings = dict(xlabel='$|\eta-1|$', ylabel='Frequency',
                    title='Distribution of $|\eta-1|$ for redshift 0.0')
    ax.set(**settings)
    kwargs = dict(markersize=0.75)
    plot_distribution(ax, abs_eta, kwargs)
    plt.draw()
    fig.savefig('plots/{}.png'.format('eta_dsitribution'))
    plt.show()

def theta_distribution(theta):
    fig, ax = plt.subplots()
    settings = dict(xlabel='$\Theta$', ylabel='Frequency',
                    title='Distribution of $\Theta$ for redshift 0.0')
    ax.set(**settings)
    kwargs = dict(markersize=0.75)
    plot_distribution(ax, theta, kwargs)
    plt.draw()
    fig.savefig('plots/{}.png'.format('theta_distribution'))
    plt.show()

def relaxation_distribution(theta, theta_err, eta):
    fig, ax = plt.subplots()
    settings = dict(xlabel='R', ylabel='Frequency',
                    title='Distribution of Relaxation Parameters for very low \u03B1 values')
    ax.set(**settings)

    for alpha in np.linspace(0.0, 0.25, 5):
        r, r_err = calc_relax(theta, theta_err, eta, alpha)
        kwargs = dict(label='\u03B1 = {}'.format(alpha), markersize=0.75)
        plot_distribution(ax, r, kwargs)

    ax.legend()
    plt.draw()
    fig.savefig('plots/{}.png'.format('r_distribution_vvvlow'))
    plt.show()

def check_relaxations(R200, alpha):
    fig, ax = plt.subplots()
    settings = dict(xlabel='R', ylabel='Frequency',
                    title='Distribution of Relaxation Parameters at different redshifts\n\u03B1 = {}'.format(alpha))
    ax.set(**settings)

    for rshift in sorted(R200.keys()):
        random_number = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        if random_number > 0:
            params = R200[rshift]
            #: ndarrays: Parameters from values.
            eta, delta, fm = params[:,1], params[:,2], params[:,3]
            #: ndarrays: Theta values and uncertainties.
            theta, theta_err = calc_theta(fm, delta)
            r, r_err = calc_relax(theta, theta_err, eta, alpha)
            kwargs = dict(label='Redshift = {}'.format(rshift), markersize=0.75)
            plot_distribution(ax, r, kwargs)
        print(R200[rshift].shape)
    plt.draw()
    fig.savefig('plots/{}.png'.format('relaxation_distribution'))
    plt.show()

def relaxation_galaxy(R200):
    galaxy_idx = 0
    alpha = 0.5
    relaxation_cols = []

    for rshift in sorted(R200.keys()):
        #: Add the theta vals onto the parameters
        params = R200[rshift]
        eta, delta, fm = params[:,1], params[:,2], params[:,3]
        theta, theta_err = calc_theta(fm, delta)
        params = np.column_stack((params, np.column_stack((theta, theta_err))))
        #: Calculate relaxation parameter for target galaxy.
        target_galaxy = params[galaxy_idx,:]
        r_param, r_param_err = calc_relax(target_galaxy[4], target_galaxy[5],
                                      target_galaxy[1], alpha)
        relaxation_cols.append([rshift, r_param, r_param_err])

    #: Turn into numpy array for easier access.
    relaxation_cols = np.array(relaxation_cols)
    rshifts, r_params, r_errs = relaxation_cols[:,0], relaxation_cols[:,1], relaxation_cols[:,2]

    #: Plot a graph of relaxation as a function of reshift.
    fig, ax = plt.subplots()
    point_settings = dict(fmt='o', markersize=2, linestyle='None', color='black')
    line_settings = dict(markersize=0.75)
    points = ax.errorbar(rshifts, r_params,yerr=r_errs, **point_settings)
    xgrid = np.linspace(np.amin(rshifts), np.amax(rshifts), 10000)
    spl = make_interp_spline(rshifts, r_params, k=3)
    power_smooth = spl(xgrid)
    line = ax.plot(xgrid, power_smooth, **line_settings)
    settings = dict(xlabel='Redshift', ylabel='R',
                    title='Relaxation against Redshift for cluster {}'.format(galaxy_idx),
                    xlim=ax.get_xlim()[::-1])
    ax.set(**settings)
    plt.draw()
    fig.savefig('plots/{}.png'.format('relaxation_cluster_{}'.format(galaxy_idx)))
    plt.show()

def main():
    #: dicts: Dictionarys of the R200 and R500 data.
    R200, R500 = read_data('dat/')
    # relaxation_galaxy(R200)
    eta = R200[0.0][:,1]
    eta_distribution(eta)

if __name__ == '__main__':
    main()
