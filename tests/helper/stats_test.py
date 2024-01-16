import warnings

from numpy import arange, isclose, isnan, nan

from pymmunomics.helper.stats import median_difference

class TestMedianDifference:
    def test_simple_arrays(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        expected = 3
        actual = median_difference(a, b)
        assert actual == expected

    def test_empty_arrays(self):
        a = []
        b = []
        expected = nan  # When arrays are empty, the result should be NaN
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            actual = median_difference(a, b)
        assert isnan(actual)

    def test_equal_arrays(self):
        a = [1, 2, 3]
        b = [1, 2, 3]
        expected = 0  # When arrays are equal, the result should be 0
        actual = median_difference(a, b)
        assert actual == expected

    def test_different_lengths(self):
        a = [1, 2, 3]
        b = [4, 5, 6, 7]
        expected = 3.5
        actual = median_difference(a, b)
        assert actual == expected

    def test_negative_values(self):
        a = [-3, -2, -1]
        b = [1, 2, 3]
        expected = 4
        actual = median_difference(a, b)
        assert actual == expected
