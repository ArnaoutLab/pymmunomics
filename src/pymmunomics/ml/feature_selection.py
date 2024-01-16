from abc import ABC, abstractmethod
from functools import partial
from math import ceil
from multiprocessing import cpu_count
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

from numpy import arange, array, concatenate, empty, isin
from numpy.typing import ArrayLike
from pandas import concat, DataFrame, isna, MultiIndex, Series
from ray import remote, get, put
from scipy.stats import kendalltau
from scipy.stats.mstats import mquantiles
from sklearn.base import BaseEstimator, TransformerMixin
from skopt.space.space import Dimension

from pymmunomics.helper.exception import InvalidArgumentError

def _kendalltau(x, y):
    correlation = kendalltau(x, y, variant="c")[0]
    if isna(correlation):
        return 0.0
    else:
        return correlation

class IdentityTransformer(TransformerMixin):
    def __init__(self, copy: bool = False):
        self.copy = copy
    def fit(self, X, y):
        return self
    def transform(self, X):
        if self.copy:
            return X.copy()
        else:
            return X

class FlattenColumnTransformer(TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
    def fit(self, X, y):
        self.transformer.fit(X, y)
        return self
    def transform(self, X):
        result = self.transformer.transform(X)
        if hasattr(result, "columns") and type(result.columns) == MultiIndex:
            flat_names = str(tuple(result.columns.names))
            flat_columns = result.columns.to_flat_index().map(str)
            result.columns = flat_columns
            result.columns.name = flat_names
        return result

@remote
def _calculate_scores_chunk(
    X, train_index, chunk_index, chunk_size, score_func, y,
):
    # train_index_chunk = train_index[chunk_index:chunk_index + chunk_size]
    chunk_stop = min(chunk_index + chunk_size, X.shape[1])
    X_train_chunk = X.to_numpy()[train_index, chunk_index:chunk_stop].T
    out_chunk = empty(X_train_chunk.shape[0])
    y_train = y[train_index]
    for j, col in enumerate(X_train_chunk):
        out_chunk[j] = score_func(col, y_train)
    return out_chunk

class NullScoreSelectorBase(BaseEstimator, TransformerMixin, ABC):
    def __init__(
        self,
        null_X: DataFrame,
        null_y: ArrayLike,
        train_X: Union[DataFrame, None] = None,
        train_y: Union[ArrayLike, None] = None,
        score_func: Callable = _kendalltau,
        alpha: float = 0.05,
        flatten_columns: bool = False,
    ):
        """Base class for variable selectors via null distribution of scores.

        Parameters
        ----------
        null_X, train_X, null_y, train_y:
            Rows correspond to samples, columns to variables for
            `null_X` and `train_X`. Index and column headers are used to
            identify samples and variables, respectively. `null_y`, and
            `train_y` are the dependent variable values. Either both, or
            neither of `train_X` and `train_y` must be provided.
        score_func:
            Function that takes two numpy arrays and returns a score.
            Will be used to assign scores to variables in null and
            train data. Called with one column of null or train data as
            the first argument and the dependent variable column as the
            second. Default calculates Kendall-tau-C correlation
            coefficient, but returns 0 when ``scipy.stats.kendalltau`
            productes a NaN result.
        alpha:
            Determines how extreme selected variable scores have to be
            compared to null scores.
        flatten_columns:
            If True and the columns of null_X (train_X) are in a
            pandas.MultinIndex, the index will be flattened. This might
            be necessary for subsequent scikit-learn functions to be
            able to accept the transformed X as input.
        """
        self.null_X = null_X
        self.null_y = null_y
        self.train_X = train_X
        self.train_y = train_y
        self.score_func = score_func
        self.alpha = alpha
        self.flatten_columns = flatten_columns

        self.null_scores = empty(self.null_X.shape[1])
        self.train_scores = None
        self.lower_quantile = None
        self.upper_quantile = None
        self.selected_columns = None

    @abstractmethod
    def fit(self, X, y):
        pass

    def transform(self, X):
        """Filters `X` returning only the selected columns in `X`."""
        result = X[self.selected_columns]
        if self.flatten_columns and type(result.columns) == MultiIndex:
            flat_names = str(tuple(result.columns.names))
            flat_columns = result.columns.to_flat_index().map(str)
            result.columns = flat_columns
            result.columns.name = flat_names
        return result

    def _decide_train_Xy(self, X, y):
        if self.train_X is None:
            self.train_X, self.train_y = X, y

    def _calculate_scores(self, train_index_items):
        self.train_scores = empty(self.train_X.shape[1])
        for full_index, X, y, out in [
            (self.null_X.index, self.null_X, self.null_y, self.null_scores),
            (self.train_X.index, self.train_X, self.train_y, self.train_scores),
        ]:
            train_index = full_index.isin(train_index_items)
            chunk_size = ceil(X.shape[1] / cpu_count())
            X_ref = put(X)
            train_index_ref = put(train_index)
            # out_ref = put(out)
            y_ref = put(y)
            chunks = []
            for chunk_index in range(0, X.shape[1], chunk_size):
                chunk_future = _calculate_scores_chunk.remote(
                    X=X_ref,
                    train_index=train_index_ref,
                    chunk_index=chunk_index,
                    chunk_size=chunk_size,
                    # out=out_ref,
                    score_func=self.score_func,
                    y=y_ref,
                )
                chunks.append(chunk_future)
            out[:] = concatenate(get(chunks))
            # train_index, chunk_index, chunk_size, out, score_func, y,
            # for j, col in enumerate(X.to_numpy()[train_index].T):
            #     out[j] = self.score_func(col, y[train_index])

class SelectNullScoreOutlier(NullScoreSelectorBase):
    def fit(self, X: DataFrame, y: ArrayLike):
        """Calculates scores and determines outlying variables.

        Selected variables are those whose score is below the
        `alpha / 2`'th or above the `1 - (alpha / 2)`'th quantile of
        null scores.

        Parameters
        ----------
        X:
            The train data from which to select outlying independent
            variables. The index will identify which samples in the null
            data to use for generating the null score distribution.
        y:
            The dependent variable's values corresponding in order to
            the train data in `X`.

        Notes
        -----
        - If `train_data` and `train_y` were passed as arguments
          to the initializer, arguments `X` and `y` are ignored, except
          for determining which samples are to be used.

        Returns
        -------
        self:
            Fitted feature-selector.
        """
        super().fit(X, y)
        self._decide_train_Xy(X, y)
        self._calculate_scores(set(X.index))
        self.lower_quantile, self.upper_quantile = mquantiles(
            self.null_scores,
            prob=[self.alpha / 2, 1 - (self.alpha / 2)],
        )
        self.selected_columns = self.train_X.columns[
            (self.train_scores < self.lower_quantile)
            | (self.upper_quantile < self.train_scores)
        ]
        return self

class SelectPairedNullScoreOutlier(NullScoreSelectorBase):
    def fit(self, X, y):
        """Calculates scores and determines outlying variables.

        Selected variables are those whose difference in null and
        train data score is below the `alpha / 2`'th or above the
        `1 - (alpha / 2)`'th quantile of all score differences.

        Parameters
        ----------
        X:
            The train data from which to select outlying independent
            variables. The index will identify which samples in the null
            data to use for generating the null score distribution.
        y:
            The dependent variable's values corresponding in order to
            the train data in `X`.

        Notes
        -----
        - If `train_data` and `train_y` were passed as arguments
          to the initializer, arguments `X` and `y` are ignored, except
          for determining which samples are to be used.

        Returns
        -------
        self:
            Fitted feature-selector.
        """
        super().fit(X, y)
        self._decide_train_Xy(X, y)
        self.train_X = self.train_X[self.null_X.columns]
        self._calculate_scores(set(X.index))
        delta_scores = self.train_scores - self.null_scores
        self.lower_quantile, self.upper_quantile = mquantiles(
            delta_scores,
            prob=[self.alpha / 2, 1 - (self.alpha / 2)],
        )
        self.selected_columns = self.train_X.columns[
            (delta_scores < self.lower_quantile)
            | (self.upper_quantile < delta_scores)
        ]
        return self

class AggregateNullScoreOutlier(NullScoreSelectorBase):
    def fit(self, X: DataFrame, y: ArrayLike):
        """Calculates scores and determines outlying variables.

        Selected variables are those whose score is below the
        `alpha / 2`'th or above the `1 - (alpha / 2)`'th quantile of
        null scores.

        Parameters
        ----------
        X : DataFrame
            The train data from which to select outlying independent
            variables. The index will identify which samples in the null
            data to use for generating the null score distribution.
        y : ArrayLike
            The dependent variable's values corresponding in order to
            the train data in `X`.

        Notes
        -----
        - If `train_data` and `train_y` were passed as arguments
          to the initializer, arguments `X` and `y` are ignored, except
          for determining which samples are to be used.

        Returns
        -------
        self:
            Fitted feature-selector.
        """
        super().fit(X, y)
        self._decide_train_Xy(X, y)
        self._calculate_scores(set(X.index))
        self.lower_quantile, self.upper_quantile = mquantiles(
            self.null_scores,
            prob=[self.alpha / 2, 1 - (self.alpha / 2)],
        )
        self.selected_columns = self.train_X.columns[
            (self.train_scores < self.lower_quantile)
            | (self.upper_quantile < self.train_scores)
        ]
        return self

    def transform(self, X):
        """Filters `X` returning only the selected columns in `X`."""
        return (
            X[self.selected_columns]
            .sum(axis=1)
            .to_frame()
            .rename(columns={
                0: ";".join(map(str, self.selected_columns.to_list()))
            })
        )
