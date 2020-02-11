"""Script to create the relaxation parameter.

"""
from utils import *

def main():
    #: dicts: Dictionarys of the R200 and R500 data.
    R200, R500 = read_data('dat/')
    for rshift, params in R200.items():
        #: ndarrays: Parameters from values.
        eta, delta, fm = params[:,1], params[:,2], params[:,3]
        #: ndarrays: Theta values and uncertainties.
        theta, theta_err = calc_theta(fm, delta)
        #: Add these columns to the ndarray.
        params = np.concatenate([
                                 np.concatenate([params,theta[:,None]],axis=1),
                                 theta_err[:,None]],
                                 axis=1)

if __name__ == '__main__':
    main()
