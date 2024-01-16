from itertools import product
from multiprocessing import cpu_count, Pool
from typing import Callable, Sequence, Tuple

from numpy import abs as np_abs, argmax, array, median, sum as np_sum
from numpy.typing import ArrayLike
from pandas import DataFrame
from scipy.stats import mannwhitneyu

from pymmunomics.helper.exception import InvalidArgumentError

def mannwhitneyu_pvalue(a, b, **kwargs):
    return mannwhitneyu(a, b, **kwargs).pvalue

# def _validate_compare_cumulatives_at_best_separator_args(
#     x: ArrayLike, y1: ArrayLike, y2: ArrayLike,
# ):
#     """
#     Validates arguments for the function
#     `compare_cumulatives_at_best_separator`.
#     """
#     if len(y1.shape) != 2 or len(y2.shape) != 2 or y1.shape[0] != y2.shape[0]:
#         raise InvalidArgumentError(
#             "y1 and y2 must be 2-d arrays with same first dimension"
#         )
#     if len(x.shape) != 1 or x.shape[0] != y1.shape[0]:
#         raise InvalidArgumentError(
#             "x must be a 1-d array with first dimension equal to that"
#             " of y1 and y2"
#         )

def median_difference(a: ArrayLike, b: ArrayLike):
    """Returns median difference of items from b minus items from a.

    Parameters
    ----------
    a, b:
        Values to take differences of.

    Returns
    -------
    s:
        Median difference of items from b minus items from a.
    """
    theilsen_estimator = median(
        [b_ - a_ for a_, b_ in product(a, b)]
    )
    return theilsen_estimator

# def compare_cumulatives_at_best_separator(
#     x: ArrayLike,
#     y1: ArrayLike,
#     y2: ArrayLike,
#     slope_func: Callable = median_difference,
#     test_func: Callable = mannwhitneyu_pvalue,
# ):
#     x, y1, y2 = map(array, [x, y1, y2])
#     _validate_compare_cumulatives_at_best_separator_args(x, y1, y2)

#     with Pool(cpu_count()) as pool:
#         slopes = pool.starmap(slope_func, zip(y1, y2))
#     threshold_sums = array([
#         np_abs(np_sum(slopes, where=(x <= t)))
#         + np_abs(np_sum(slopes, where=(x > t)))
#         for t in x
#     ])
#     separator = x[argmax(threshold_sums)]
#     left_index = x <= separator
#     if np_sum(slopes, where=left_index) > 0:
#         alternative = "less"
#     else:
#         alternative = "greater"
#     y1_cumulatives, y2_cumulatives = map(
#         lambda y: np_sum(y, where=left_index.reshape(-1,1), axis=0),
#         [y1, y2],
#     )
#     pvalue = test_func(y1_cumulatives, y2_cumulatives, alternative=alternative)
#     return (pvalue, slopes, separator, (y1_cumulatives, y2_cumulatives), alternative)
