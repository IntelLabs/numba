import numba
import numpy as np
import h5py
import argparse
import time

@numba.njit(parallel=True, locals={'X':numba.float64[:,:], 'Y':numba.float64[:]})
def logistic_regression(file_name, iterations):
    f = h5py.File(file_name, "r")
    X = f['points'][:]
    Y = f['responses'][:]
    D = X.shape[1]
    w = 2.0*np.ones(D)-1.3
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y),X)
    return w

def main():
    parser = argparse.ArgumentParser(description='Logistic Regression.')
    parser.add_argument('--file', dest='file', type=str, default="lr.hdf5")
    parser.add_argument('--iterations', dest='iterations', type=int, default=30)
    args = parser.parse_args()

    file_name = args.file
    iterations = args.iterations

    t = time.time()
    w = logistic_regression(file_name, iterations)
    selftimed = time.time()-t
    print("SELFTIMED ", selftimed)
    print("result: ", w)

if __name__ == '__main__':
    main()
