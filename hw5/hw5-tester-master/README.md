# hw5-tester

Tests for CS540 Spring 2021 HW5: Linear Regression

## Changes

### V1.3
 - display which values were used when tests fail/have errors
 - add more tests for `print_stats`

### V1.2
 - add more tests for `synthetic_datasets`

### V1.1.1
 - check quadratic dataset std in addition to linear std
 - current version should format correctly if out of date

### V1.1
 - test that `plot_mse` generates `mse.pdf` (can't check graph output)

## Usage

Download [test.py](test.py) move it into the directory that contains `regression.py` and `bodyfat.csv`

The contents of your directory should look like this:

```shell
$ tree
.
├── regression.py
├── bodyfat.csv
└── test.py
```

To run the tests, do

```python
$ python3 test.py
```

Due to some of the tests needing to check the printed output of your function, you may have issues like `"AttributeError: '_io.TextIOWrapper' object has no attribute 'getvalue'"` if you don't run the tester in your terminal.

### These tests _do not_ check if `plot_mse` has a correct graph, only that it generates `mse.pdf`

## Disclaimer

These tests are not endorsed or created by anyone working in an official capacity with UW Madison or any staff for CS540. The tests are make by students, for students.

By running `test.py`, you are executing code you downloaded from the internet. Back up your files and take a look at what you are running first.

If you have comments or questions, create an issue at [https://github.com/CS540-testers-SP21/hw3-tester/issues](https://github.com/CS540-testers-SP21/hw4-tester/issues) or ask in our discord at [https://discord.gg/RDFNsAxgCQ](https://discord.gg/RDFNsAxgCQ).
