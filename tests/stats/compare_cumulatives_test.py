from numpy import allclose, array, isclose
from pytest import fixture, raises

from pymmunomics.helper.exception import InvalidArgumentError
from pymmunomics.stats.compare_cumulatives import (
    compare_best_slope_separator_cumulatives,
    get_best_separator_pos,
)

class TestGetBestSeparatorPos:
    def test_simple(self):
        items = array([-1, -2, -3, 4, 5])
        expected = 3
        actual = get_best_separator_pos(items)
        assert actual == expected

    def test_ambiguous(self):
        items = array([1, 2, 3, 4, 5])
        expected = {0, 1, 2, 3, 4} # any position will do
        actual = get_best_separator_pos(items)
        assert actual in expected

    def test_empty_array(self):
        items = array([])
        with raises(InvalidArgumentError):
            get_best_separator_pos(items)

    def test_multiple_dimensions(self):
        items = array([[1, 2, 3], [4, 5, 6]])
        with raises(InvalidArgumentError):
            get_best_separator_pos(items)

    def test_complex(self):
        items = array([-1, -2, 3, -4, 5, 6,])
        expected = 4
        actual = get_best_separator_pos(items)
        assert actual == expected

class TestCompareBestSlopeSeparatorCumulatives:
    @fixture
    def a(self):
        return array([
            [0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
            [0.5, 0.6, 0.7, 0.2, 0.3, 0.4],
            [0.6, 0.7, 0.8, 0.3, 0.4, 0.5],
        ])

    @fixture
    def b(self):
        return array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ])

    def test_simple(self, a, b):
        actual = compare_best_slope_separator_cumulatives(a, b)
        assert isclose(actual.pvalue, 0.1)
        assert allclose(actual.cumulatives[0], array([1.5, 1.8, 2.1]))
        assert allclose(actual.cumulatives[1], array([0.6, 0.9, 1.2]))
        assert allclose(actual.slopes, array([-0.3, -0.3, -0.3, 0.3, 0.3, 0.3]))
        assert actual.best_separator_pos == 3

    def test_callable_slope(self, a, b):
        slope = lambda x, y: y[2] - x[0]
        actual = compare_best_slope_separator_cumulatives(a, b, slope=slope)
        assert isclose(actual.pvalue, 0.1)
        assert allclose(actual.cumulatives[0], array([1.5, 1.8, 2.1]))
        assert allclose(actual.cumulatives[1], array([0.6, 0.9, 1.2]))
        assert allclose(actual.slopes, array([-0.1, -0.1, -0.1, 0.5, 0.5, 0.5]))
        assert actual.best_separator_pos == 3

    def test_fixed_slopes(self, a, b):
        slope = array([-0.2, -0.1, 0.3, 0.4, 0.5, 0.6])
        actual = compare_best_slope_separator_cumulatives(a, b, slope=slope)
        assert isclose(actual.pvalue, 0.1)
        assert allclose(actual.cumulatives[0], array([0.9, 1.1, 1.3]))
        assert allclose(actual.cumulatives[1], array([0.3, 0.5, 0.7]))
        assert allclose(actual.slopes, slope)
        assert actual.best_separator_pos == 2

    def test_test_func(self, a, b):
        test_func = lambda x, y: x.sum() + y.sum()
        actual = compare_best_slope_separator_cumulatives(a, b, test_func=test_func)
        assert isclose(actual.pvalue, 8.1)
        assert allclose(actual.cumulatives[0], array([1.5, 1.8, 2.1]))
        assert allclose(actual.cumulatives[1], array([0.6, 0.9, 1.2]))
        assert allclose(actual.slopes, array([-0.3, -0.3, -0.3, 0.3, 0.3, 0.3]))
        assert actual.best_separator_pos == 3

    def test_wrong_shape(self, a, b):
        with raises(InvalidArgumentError):
            compare_best_slope_separator_cumulatives(a[0], b[0])

    def test_inconsistent_column_numbers(self, a, b):
        with raises(InvalidArgumentError):
            compare_best_slope_separator_cumulatives(a[:,:5], b)

    def test_empty_a(self, a, b):
        with raises(InvalidArgumentError):
            compare_best_slope_separator_cumulatives(a[:0], b)

    def test_empty_b(self, a, b):
        with raises(InvalidArgumentError):
            compare_best_slope_separator_cumulatives(a, b[:0])

    def test_wrong_fixed_slope_shape(self, a, b):
        slope = array([[-0.2, -0.1, 0.3, 0.4, 0.5, 0.6]])
        with raises(InvalidArgumentError):
            compare_best_slope_separator_cumulatives(a, b, slope=slope)

    def test_inconsistent_fixed_slope_number(self, a, b):
        slope = array([-0.2, -0.1, 0.3, 0.4, 0.5])
        with raises(InvalidArgumentError):
            compare_best_slope_separator_cumulatives(a, b, slope=slope)
