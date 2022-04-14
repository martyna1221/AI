'''
These tests are inspired by and use code from the tests made by cs540-testers
for the Fall 2020 semester

Their version can be found here: https://github.com/cs540-testers/hw5-tester/
'''

__maintainer__ = 'CS540-testers-SP21'
__author__ = ['Nicholas Beninato']
__credits__ = ['Harrison Clark', 'Stephen Jasina', 'Saurabh Kulkarni', 'Alex Moon']
__version__ = '1.0'

import sys
import unittest
from time import time
import numpy as np
from pca import *

file_path = 'YaleB_32x32.npy'

def timeit(func):
    def timed_func(*args, **kwargs):
        t0 = time()
        out = func(*args, **kwargs)
        print(f'Ran {func.__name__}{" "*(30-len(func.__name__))}in {(time() - t0)*1000:.2f}ms')
        return out
    return timed_func

class TestPrincipalComponentAnalysis(unittest.TestCase):
    @timeit
    def test1_load_and_center_dataset(self):
        X = load_and_center_dataset(file_path)

        # Dataset needs to have the correct shape
        self.assertEqual(X.shape, (2414, 1024))

        # The mean should be close to 0 (account for floating point math)
        self.assertTrue(np.isclose(X.mean(), 0.0))

    @timeit
    def test2_get_covariance(self):
        X = load_and_center_dataset(file_path)
        S = get_covariance(X)

        # S needs have size d x d
        self.assertEqual(S.shape, (1024, 1024))

        # S should be symmetric
        self.assertTrue(np.allclose(S, S.T))

        # S should have non-negative values on the diagonal
        self.assertTrue(np.min(np.diagonal(S)) >= 0)

    @timeit
    def test3_get_eig_small(self):
        X = load_and_center_dataset(file_path)
        S = get_covariance(X)

        Lambda, U = get_eig(S, 2)

        # Eigenvalues need to have shape 2 2
        self.assertEqual(Lambda.shape, (2, 2))

        # Eigenvalues should match example
        self.assertTrue(np.allclose(Lambda, [[1369142.41612494, 0],[0, 1341168.50476773]]))

        # Eigenvectors need to have shape 2 2
        self.assertEqual(U.shape, (1024, 2))

        # Av = λv (matrix * vector = scalar * vector)
        self.assertTrue(np.allclose(S @ U, U @ Lambda))

    @timeit
    def test4_get_eig_large(self):
        X = load_and_center_dataset(file_path)
        S = get_covariance(X)
        Lambda, U = get_eig(S, 1024)

        # Eigenvalues need to have shape 1024 1024
        self.assertEqual(np.shape(Lambda), (1024, 1024))
        
        # Check that Lambda is diagonal
        self.assertEqual(np.count_nonzero(
                         Lambda - np.diag(np.diagonal(Lambda))), 0)
        
        # Check that Lambda is sorted in decreasing order
        diag = np.diagonal(Lambda)
        self.assertTrue(np.all(np.equal(diag, diag[np.argsort(-diag)])))

        # Eigenvectors need to have shape 1024 1024
        self.assertEqual(np.shape(U), (1024, 1024))

        # Av = λv (matrix * vector = scalar * vector)
        self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

    @timeit
    def test5_get_eig_perc_small(self):
        X = load_and_center_dataset(file_path)
        S = get_covariance(X)

        Lambda, U = get_eig_perc(S, 0.07)

        # Eigenvalues need to have shape 2 2
        self.assertEqual(Lambda.shape, (2, 2))

        # Eigenvalues should match example
        self.assertTrue(np.allclose(Lambda, [[1369142.41612494, 0],[0, 1341168.50476773]]))

        # Eigenvectors need to have shape 2 2
        self.assertEqual(U.shape, (1024, 2))

        # Av = λv (matrix * vector = scalar * vector)
        self.assertTrue(np.allclose(S @ U, U @ Lambda))

    @timeit
    def test6_get_eig_perc_large(self):
        X = load_and_center_dataset(file_path)
        S = get_covariance(X)
        Lambda, U = get_eig_perc(S, -1)

        # Eigenvalues need to have shape 1024 1024
        self.assertEqual(np.shape(Lambda), (1024, 1024))
        
        # Check that Lambda is diagonal
        self.assertEqual(np.count_nonzero(
                         Lambda - np.diag(np.diagonal(Lambda))), 0)
        
        # Check that Lambda is sorted in decreasing order
        diag = np.diagonal(Lambda)
        self.assertTrue(np.all(np.equal(diag, diag[np.argsort(-diag)])))

        # Eigenvectors need to have shape 1024 1024
        self.assertEqual(np.shape(U), (1024, 1024))

        # Av = λv (matrix * vector = scalar * vector)
        self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

    @timeit
    def test7_project_image(self):
        X = load_and_center_dataset(file_path)
        S = get_covariance(X)
        Lambda, U = get_eig(S, 2)
        projected = project_image(X[0], U)
    
        # Projected needs to have shape (1024, )
        self.assertEqual(projected.shape, (1024,))

        # Example values from Canvas
        self.assertTrue(np.allclose(projected[:3], [6.84122225,4.83901287,1.41736694]))
        self.assertTrue(np.allclose(projected[-3:], [8.75796534,7.45916035,5.4548656]))

        # Min and max values
        self.assertTrue(np.isclose(projected.max(), 93.22417310945819))
        self.assertTrue(np.isclose(projected.min(), 0.27875793275475225))

if __name__ == '__main__':
    print(f'Running CS540 SP21 HW3 tester v{__version__}')
    # Hack to allow different locations of YaleB_32x32.npy (done this way to allow
    # unittest's flags to still be passed, if desired)
    if '--yale-path' in sys.argv:
        path_index = sys.argv.index('--yale-path') + 1
        if path_index == len(sys.argv):
            print('Error: must supply path after option --yale-path')
            sys.exit(1)
        file_path = sys.argv[path_index]
        print(f'Using {file_path} as location of dataset')
        del(sys.argv[path_index])
        del(sys.argv[path_index - 1])

    unittest.main(argv=sys.argv)
