'''
These tests are inspired by and use code from the tests made by cs540-testers
for the Fall 2020 semester

Their version can be found here: https://github.com/cs540-testers/hw8-tester/
'''

__maintainer__ = 'CS540-testers-SP21'
__author__ = ['Nicholas Beninato']
__credits__ = ['Harrison Clark', 'Stephen Jasina', 'Saurabh Kulkarni', 'Alex Moon']
__version__ = '1.3'

import unittest
import io
import sys
from os import path, remove
from time import time, sleep
from urllib.request import urlopen
import numpy as np
from regression import *

failures = dict()
errors = dict()
test_output = []

BODYFAT_FILE = 'bodyfat.csv'

def timeit(func):
    # wrapper to time tests and add results to lists to display after all tests finish
    def timed_func(*args, **kwargs):
        global failures, errors
        t0 = time()
        try:
            out = func(*args, **kwargs)
            runtime = time() - t0
        except AssertionError as e: # failed test
            test_output.append(f'FAILED {func.__name__}')
            failures[func.__name__] = TestRegression.latest_test
            raise e
        except Exception as e: # error when running test
            test_output.append(f'ERROR  {func.__name__}')
            errors[func.__name__] = TestRegression.latest_test
            raise e
        test_output.append(f'PASSED {func.__name__}{" "*(26-len(func.__name__))}in {(runtime)*1000:.2f}ms')
    return timed_func

class CapturedOutput():
    # context manager to capture stdout
    def __init__(self):
        self.output = ''

    def __enter__(self):
        self.capturedOutput = io.StringIO()
        sys.stdout = self.capturedOutput
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.output = self.capturedOutput.getvalue()
        sys.stdout = sys.__stdout__

class TestRegression(unittest.TestCase):

    latest_test = {None:None}

    @timeit
    def test1_get_dataset(self):
        dataset = get_dataset(BODYFAT_FILE)
        TestRegression.latest_test = {'input':{'csv path':BODYFAT_FILE}}

        self.assertIsInstance(dataset, np.ndarray)
        self.assertEqual(dataset.shape, (252,16)) # check shape and a few values
        self.assertEqual(dataset[0][0], 12.6)
        self.assertEqual(dataset[251][0], 30.7)
        self.assertEqual(dataset[75][7], 91.6)

    @timeit
    def test2_print_stats(self):
        dataset = get_dataset(BODYFAT_FILE)

        subsets = [(None,None), (10,20), (0,100), (30,None), (99,111)] # various sized datasets
        cols = [1, 5, 2, 15, 1]
        expected_output = ["252\n1.06\n0.02\n", "10\n27.07\n1.57\n",
                           "100\n44.85\n14.24\n", "222\n18.27\n0.94\n",
                           "12\n1.05\n0.00\n"]

        for (start, end), col, expected in zip(subsets, cols, expected_output):
            TestRegression.latest_test = {'input':{'start_index':start, 'end_index':end, 'column':col}, 
                                          'output':'Test failed before output', 'expected':expected}

            with CapturedOutput() as c: # get data from stdout
                print_stats(dataset[start:end,:], col)
            TestRegression.latest_test['output'] = c.output
            self.assertEqual(c.output, expected)

    @timeit
    def test3_regression(self):
        dataset = get_dataset(BODYFAT_FILE)

        cols = [[2,3], [2,3,4], [9], [1,8,9,10,13]]
        betas = [[0,0,0], [0,-1.1,-.2,3], [-3,1], [1,0,-1,0,1,0]]
        expected_output = [418.50384920634923, 11859.17408611111,
                           6120.485198412698, 2816.662023809524]

        for c, b, expected in zip(cols, betas, expected_output):
            TestRegression.latest_test = {'input':{'columns':c, 'betas':b}, 
                                          'output':'Test failed before output', 'expected':expected}
            mse = regression(dataset, cols=c, betas=b)
            TestRegression.latest_test['output'] = mse
            self.assertTrue(np.isclose(mse, expected))

    @timeit
    def test4_gradient_descent(self):
        dataset = get_dataset(BODYFAT_FILE)

        cols = [[2,3],[4,5,7],[5,10,13]]
        betas = [[0,0,0],[90,-180,200,70],[100,-300,200,-100]]
        expected_output = [[-37.87698413, -1756.37222222, -7055.35138889],
                           [-821.29761905, -61970.46011905, -11600.26111111, -63308.49904762],
                           [2207.67857143, 52724.57619048, 130376.63777778, 69303.68420635]]
        
        for c, b, expected in zip(cols, betas, expected_output):
            TestRegression.latest_test = {'input':{'columns':c, 'betas':b}, 
                                          'output':'Test failed before output', 'expected':expected}
            grad_desc = gradient_descent(dataset, cols=c, betas=b)
            TestRegression.latest_test['output'] = grad_desc
            self.assertTrue(np.allclose(grad_desc, np.array(expected)))

    @timeit
    def test5_iterate_gradient(self):
        dataset = get_dataset(BODYFAT_FILE)

        cols = [[1,8],[2,3,4,14,15]]
        betas = [[400,-400,300],[-100,0,100,1000,0,100]]
        T = [10,12]
        eta = [1e-4,1e-5]
        expected_output = [["1 423085332.40 394.45 -405.84 -220.18",
                            "2 229744495.73 398.54 -401.54 163.14",
                            "3 124756241.68 395.53 -404.71 -119.33",
                            "4 67745350.04 397.75 -402.37 88.82",
                            "5 36787203.39 396.11 -404.09 -64.57",
                            "6 19976260.50 397.32 -402.82 48.47",
                            "7 10847555.07 396.43 -403.76 -34.83",
                            "8 5890470.68 397.09 -403.07 26.55",
                            "9 3198666.69 396.60 -403.58 -18.68",
                            "10 1736958.93 396.96 -403.20 14.65"],
                            ["1 342100904.70 -101.79 -80.40 -223.57 873.75 -51.56 67.22",
                             "2 81592526.77 -102.14 -95.57 -282.23 849.37 -61.35 60.98",
                             "3 72504356.18 -102.22 -98.78 -292.33 843.67 -63.47 59.61",
                             "4 71868040.98 -102.25 -99.80 -293.53 841.39 -64.19 59.13",
                             "5 71517760.63 -102.28 -100.40 -293.10 839.75 -64.66 58.82",
                             "6 71179059.66 -102.30 -100.93 -292.37 838.22 -65.07 58.54",
                             "7 70842699.81 -102.32 -101.45 -291.59 836.72 -65.48 58.27",
                             "8 70508356.46 -102.34 -101.95 -290.80 835.22 -65.88 57.99",
                             "9 70176004.71 -102.36 -102.46 -290.02 833.73 -66.28 57.72",
                             "10 69845629.85 -102.38 -102.95 -289.23 832.25 -66.68 57.45",
                             "11 69517217.60 -102.40 -103.45 -288.45 830.76 -67.08 57.18",
                             "12 69190753.82 -102.42 -103.94 -287.67 829.28 -67.48 56.91"]]

        for col, b, t, e, expected in zip(cols, betas, T, eta, expected_output):
            TestRegression.latest_test = {'input':{'columns':col, 'betas':b, 'steps':t, 'eta':e}, 
                                          'output':'Test failed before output', 
                                          'expected':'\n'.join(expected)}
            with CapturedOutput() as c: # get data from stdout
                iterate_gradient(dataset, cols=col, betas=b, T=t, eta=e)
            
            TestRegression.latest_test['output'] = c.output
            for out_line, exp_line in zip(c.output.split('\n'), expected):
                self.assertEqual(out_line.rstrip(), exp_line)

    @timeit
    def test6_compute_betas(self):
        dataset = get_dataset(BODYFAT_FILE)

        cols = [[1,2], [4,7,10,14], [2,4,8]]
        expected_output = [(1.4029395600144443, 441.3525943592249, -400.5954953685588, 0.009892204826346139),
                           (27.675832287038542, -20.639574843221197, -0.39359551188970887, 0.6047052815235706, 0.20692000607810693, -0.21186541143734056),
                           (18.33521468045498, -15.21736403594966, 0.04775434431696912, -0.31062670246608437, 0.5812970224503422)]
        
        for c, expected in zip(cols, expected_output):
            TestRegression.latest_test = {'input':{'columns':c}, 
                                          'output':'Test failed before output', 'expected':expected}
            betas = compute_betas(dataset, c)
            TestRegression.latest_test['output'] = betas
            self.assertTrue(np.allclose(np.array(expected), np.array(betas)))
    
    @timeit
    def test7_predict(self):
        dataset = get_dataset(BODYFAT_FILE)

        cols = [[1,2], [3,4], [2], [6,7,9,10]]
        features = [[1.0708, 23], [180,72], [50], [37.2, 99.1,90, 62.3]]
        expected_output = [12.62245862957813, 17.935385561001944, 19.848237102542832, 17.56389897468408]
        
        for c, f, expected in zip(cols, features, expected_output):
            TestRegression.latest_test = {'input':{'columns':c, 'features':f}, 
                                          'output':'Test failed before output', 'expected':expected}
            prediction = predict(dataset, cols=c, features=f)
            TestRegression.latest_test['output'] = prediction
            self.assertTrue(np.isclose(prediction, expected))

    @timeit
    def test8_synthetic_datasets(self):
        betas =  [[0,2], [0,2], [5,10], [5.5,20]]
        alphas = [[0,1], [0,1], [3,2],  [-2,  3]]
        X = [[4], [4,3,2], [0]*20, list(range(10))]
        sigma = [1, 10, 100, 1000]
        linear_output = [[8], [8,6,4], [5]*20,
                         [5.5, 25.5, 45.5, 65.5, 85.5, 105.5, 125.5, 145.5, 165.5, 185.5]]
        quadratic_output = [[16], [16,9,4], [3]*20,
                            [-2, 1, 10, 25, 46, 73, 106, 145, 190, 241]]

        for b, a, x, s, l, q in zip(betas, alphas, X, sigma, linear_output, quadratic_output):
            x = list(map(lambda v:[v], x)) # make each value of x a list
            TestRegression.latest_test = {'input':{'betas':b, 'alphas':a, 'X':x, 'sigma':s}, 
                                          'linear mean output':'Test failed before output', 
                                          'linear std output':'Test failed before output', 
                                          'linear expected (without noise)':l,
                                          'quadratic mean output':'Test failed before output',
                                          'quadratic std output':'Test failed before output',
                                          'quadratic expected (without noise)': q}
            linear, quadratic = synthetic_datasets(np.array(b), np.array(a), np.array(x), s)

            self.assertEqual(linear.shape, (len(x),2)) # n x 2 shape, col 1 should be x values
            self.assertEqual(quadratic.shape, (len(x),2))
            self.assertTrue(np.allclose(linear[:,1], np.array(x).flatten()))
            self.assertTrue(np.allclose(quadratic[:,1], np.array(x).flatten()))

            iterations = 4000
            linear_values = np.empty((iterations, len(x)))
            quadratic_values = np.empty((iterations, len(x)))
            for i in range(iterations): # many samples to find mean and standard deviation
                linear, quadratic = synthetic_datasets(np.array(b), np.array(a), np.array(x), s)
                linear_values[i] = linear[:,0]
                quadratic_values[i] = quadratic[:,0]

            TestRegression.latest_test['linear mean output'] = np.array(linear_values).mean(axis=0)
            TestRegression.latest_test['quadratic mean output'] = np.array(quadratic_values).mean(axis=0)
            TestRegression.latest_test['linear std output'] = np.array(linear_values).std(axis=0).mean()
            TestRegression.latest_test['quadratic std output'] = np.array(quadratic_values).std(axis=0).mean()
            self.assertTrue(np.allclose(np.array(linear_values).mean(axis=0), np.array(l), atol=s/10))
            self.assertTrue(np.allclose(np.array(quadratic_values).mean(axis=0), np.array(q), atol=s/10))
            self.assertTrue(np.isclose(np.array(linear_values).std(axis=0).mean(), s, atol=s/10))
            self.assertTrue(np.isclose(np.array(quadratic_values).std(axis=0).mean(), s, atol=s/10))

    @timeit
    def test9_plot_mse(self):
        TestRegression.latest_test = {None:None}
        if path.isfile('mse.pdf'): # delete file if it exists
            remove('mse.pdf')
        plot_mse()
        self.assertTrue(path.isfile('mse.pdf')) # check if file exists after running

def display_issues(issues):
    for name, details in issues.items():
        print(name + '\n')
        for title, data in details.items():
            if title is None:
                continue
            print(f'****{title}****')
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f'{k}: {v}')
            else:
                print(f'{data}')
    print()

def get_versions():
    current = __version__
    to_tuple = lambda x: tuple(map(int, x.split('.')))
    try:
        with urlopen('https://raw.githubusercontent.com/CS540-testers-SP21/hw5-tester/master/.version') as f:
            if f.status != 200:
                raise Exception
            latest = f.read().decode('utf-8')
    except Exception as e:
        print('Erorr checking for latest version') # very descriptive error messages
        return to_tuple(current), to_tuple(current) # ignoring errors probably isn't the best idea tbh
    return to_tuple(current), to_tuple(latest)

if __name__ == '__main__':
    print(f'Running CS540 SP21 HW5 tester v{__version__}\n')

    current, latest = get_versions()
    to_v_str = lambda x : '.'.join(map(str, x))
    if current < latest:
        print(f'A newer version of this tester (v{to_v_str(latest)}) is available. ' +
              f'You are current running v{to_v_str(current)}\n')
        print('You can download the latest version at https://github.com/CS540-testers-SP21/hw5-tester\n')
    
    unittest.main(argv=sys.argv, exit=False)
    sleep(.01)
    for message in test_output:
        print(message)
    print()
    if not failures and not errors:
        print('\nPassed all tests successfully\n')
    if failures:
        print('The following tests failed:\n')
        display_issues(failures)
        
        if ['test8_synthetic_datasets'] == failures:
            print('Note: synthetic_datasets generates random data. If you pass all tests but ' + 
                  'occasionally fail test8_synthetic_datasets, you should be fine. ' +
                  'In order to account for the noisy data, synthetic_datasets is tested many times, ' +
                  'but it still may occasionally fail.\n')
    if errors:
        print('The following tests had exceptions when running:\n')
        display_issues(errors)

    if failures or errors:
        print('Please see the Traceback above for where there were issues')
