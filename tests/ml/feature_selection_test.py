from unittest.mock import MagicMock

from numpy import array, array_equal, nan, zeros
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from pytest import fixture
from sklearn.base import TransformerMixin

from pymmunomics.ml.feature_selection import (
    _kendalltau,
    AggregateNullScoreOutlier,
    IdentityTransformer,
    FlattenColumnTransformer,
    SelectNullScoreOutlier,
    SelectPairedNullScoreOutlier,
)

class Test_Kendalltau:
    def test_call_args(self, monkeypatch):
        with monkeypatch.context() as m:
            fake_kendalltau = MagicMock(side_effect=[("first", "second")])
            m.setattr(
                "pymmunomics.ml.feature_selection.kendalltau",
                fake_kendalltau,
            )
            _kendalltau("foo", "bar")
            fake_kendalltau.assert_called_with("foo", "bar", variant="c")

    def test_value(self, monkeypatch):
        with monkeypatch.context() as m:
            fake_kendalltau = MagicMock(side_effect=[("first", "second")])
            m.setattr(
                "pymmunomics.ml.feature_selection.kendalltau",
                fake_kendalltau,
            )
            actual_result = _kendalltau("foo", "bar")
            assert actual_result == "first"

    def test_nan(self, monkeypatch):
        with monkeypatch.context() as m:
            fake_kendalltau = MagicMock(side_effect=[(nan, "second")])
            m.setattr(
                "pymmunomics.ml.feature_selection.kendalltau",
                fake_kendalltau,
            )
            actual_result = _kendalltau("foo", "bar")
            assert actual_result == 0.0

class TestIdentityTransformer:
    def test_fit_transform_with_copy(self):
        X = array([[1,2],[3,4]])
        y = array([1,0])
        model = IdentityTransformer(copy=True).fit(X=X, y=y)
        actual_transformed = model.transform(X)
        assert array_equal(actual_transformed, X)
        assert not (actual_transformed is X)

    def test_fit_transform_without_copy(self):
        X = array([[1,2],[3,4]])
        y = array([1,0])
        model = IdentityTransformer(copy=False).fit(X=X, y=y)
        actual_transformed = model.transform(X)
        assert actual_transformed is X

class TestFlattenColumnTransformer:
    class PlusOneTransformer(TransformerMixin):
        def fit(self, X, y):
            return
        def transform(self, X):
            return X + 1

    def test_multiindex_columns(self):
        X = DataFrame(
            columns=MultiIndex.from_arrays(
                names=["i1", "i2"],
                arrays=[
                    ["i1_1", "i1_2", "i1_3"],
                    ["i2_1", "i2_2", "i2_3"],
                ],
            ),
            data=[
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [0, 1, 2],
            ]
        )
        y = array([1, 2, 3, 4])
        transformer = (
            FlattenColumnTransformer(
                TestFlattenColumnTransformer.PlusOneTransformer()
            )
            .fit(X=X, y=y)
        )
        expected_X_transformed = DataFrame(
            columns=Index(
                name=str(("i1", "i2")),
                data=[
                    str(("i1_1", "i2_1")),
                    str(("i1_2", "i2_2")),
                    str(("i1_3", "i2_3")),
                ]
            ),
            data=[
                [2, 3,  4],
                [5, 6,  7],
                [8, 9, 10],
                [1, 2,  3],
            ]
        )
        actual_X_transformed = transformer.transform(X)
        assert_frame_equal(actual_X_transformed, expected_X_transformed)

    def test_flat_index_columns(self):
        X = DataFrame(
            columns=Index(
                name="idx",
                data=["i1", "i2", "i3"],
            ),
            data=[
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [0, 1, 2],
            ]
        )
        y = array([1, 2, 3, 4])
        transformer = (
            FlattenColumnTransformer(
                TestFlattenColumnTransformer.PlusOneTransformer()
            )
            .fit(X=X, y=y)
        )
        expected_X_transformed = DataFrame(
            columns=Index(
                name="idx",
                data=["i1", "i2", "i3"],
            ),
            data=[
                [2, 3,  4],
                [5, 6,  7],
                [8, 9, 10],
                [1, 2,  3],
            ]
        )
        actual_X_transformed = transformer.transform(X)
        assert_frame_equal(actual_X_transformed, expected_X_transformed)

    def test_list_columns(self):
        X = DataFrame(
            columns=["i1", "i2", "i3"],
            data=[
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [0, 1, 2],
            ]
        )
        y = array([1, 2, 3, 4])
        transformer = (
            FlattenColumnTransformer(
                TestFlattenColumnTransformer.PlusOneTransformer()
            )
            .fit(X=X, y=y)
        )
        expected_X_transformed = DataFrame(
            columns=["i1", "i2", "i3"],
            data=[
                [2, 3,  4],
                [5, 6,  7],
                [8, 9, 10],
                [1, 2,  3],
            ]
        )
        actual_X_transformed = transformer.transform(X)
        assert_frame_equal(actual_X_transformed, expected_X_transformed)

    def test_array_data(self):
        X = array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 1, 2],
        ])
        y = array([1, 2, 3, 4])
        transformer = (
            FlattenColumnTransformer(
                TestFlattenColumnTransformer.PlusOneTransformer()
            )
            .fit(X=X, y=y)
        )
        expected_X_transformed = array([
            [2, 3,  4],
            [5, 6,  7],
            [8, 9, 10],
            [1, 2,  3],
        ])
        actual_X_transformed = transformer.transform(X)
        assert array_equal(actual_X_transformed, expected_X_transformed)

class TestSelectNullScoreOutlier:
    @fixture
    def score_func(self):
        def func(x, y):
            return x[0] + y[0]
        return func

    @fixture
    def null_data(self):
        data = DataFrame(
            index=[
                "sample_1",
                "sample_2",
                "sample_3",
                "sample_4",
            ],
            columns=[
                "null_1", "null_2", "null_3", "null_4", "null_5", "null_y",
            ],
            data=[
                [5, 1, 2, 3, 4,  1],
                [3, 4, 2, 1, 5,  0],
                [2, 4, 3, 1, 5, -1],
                [1, 2, 4, 3, 5,  0],
            ],
        )
        null_y = data.pop("null_y").to_numpy()
        # alpha first_train_row lower_quantile upper_quantile
        #  0.25        sample_1           2.05           5.95
        #  0.25        sample_2           1.05           4.95
        #  0.25        sample_3           0.05           3.95
        #  0.25        sample_4           1.05           4.95
        #  0.8         sample_1           3.48           4.52
        #  0.8         sample_2           2.48           3.52
        #  0.8         sample_3           1.48           2.52
        #  0.8         sample_4           2.48           3.52
        return (data, null_y)

    @fixture
    def train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_y",
            ],
            data=[
                [1, 2, 5, 4, 3, 1,   0],
                [3, 4, 5, 1, 2, 3,  -1],
                [2, 5, 3, 1, 4, 5,   1],
                [2, 5, 1, 3, 4, 4,  -1],
            ],
        )
        train_y = data.pop("train_y").to_numpy()
        # alpha train_samples lower_quantile upper_quantile selected_features
        #  0.25           all           2.05           5.95   train_1, train_2, train_6
        #  0.25       1, 2, 3           2.05           5.95   train_1, train_4, train_5, train_6
        #  0.25          2, 3           1.05           4.95   train_2, train_5, train_6
        #  0.25             3           0.05           3.95   train_2, train_3
        #  0.8            all           3.48           4.52   train_1, train_2, train_3, train_5, train_6
        #  0.8        1, 2, 3           3.48           4.52   train_1, train_2, train_4, train_5, train_6
        #  0.8           2, 3           2.48           3.52   train_2, train_3, train_4, train_5, train_6
        #  0.8              3           1.48           2.52   train_1, train_2, train_3, train_5, train_6
        return (data, train_y)

    @fixture
    def zeros_train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_y",
            ],
            data=[
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
            ],
        )
        train_y = data.pop("train_y").to_numpy()
        # would always all get selected
        return (data, train_y)

    def test_full_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
        )
        train_X, train_y = train_data
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[["train_1", "train_2", "train_6"]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_flatten_columns(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        null_X.columns = MultiIndex.from_arrays(
            names=[f"level_{j}" for j in range(2)],
            arrays=[
                [f"level_{j}_col_{i}" for i in range(null_X.shape[1])]
                for j in range(2)
            ]
        )
        selector = SelectNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
            flatten_columns=True,
        )
        train_X, train_y = train_data
        train_X.columns = MultiIndex.from_arrays(
            names=[f"level_{j}" for j in range(2)],
            arrays=[
                [f"level_{j}_col_{i}" for i in range(train_X.shape[1])]
                for j in range(2)
            ]
        )
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X.copy()
        expected.columns = expected.columns.to_flat_index().map(str)
        expected.columns.name = str(("level_0", "level_1"))
        expected = expected[[
            str(("level_0_col_0", "level_1_col_0")),
            str(("level_0_col_1", "level_1_col_1")),
            str(("level_0_col_5", "level_1_col_5")),
        ]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_subset_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
        )
        train_X, train_y = train_data
        train_X = train_X.iloc[1:] # subsetting train data
        train_y = train_y[1:]
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[["train_1", "train_4", "train_5", "train_6"]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_full_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        null_X, null_y = null_data
        train_X, train_y = train_data
        selector = SelectNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            train_X=train_X,
            train_y=train_y,
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_X, zeros_train_y = zeros_train_data
        selector.fit(
            X=zeros_train_X, y=zeros_train_y,
        )
        expected = zeros_train_X[["train_1", "train_2", "train_6"]]
        actual = selector.transform(zeros_train_X)
        assert_frame_equal(actual, expected)

    def test_subset_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        null_X, null_y = null_data
        train_X, train_y = train_data
        selector = SelectNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            train_X=train_X,
            train_y=train_y,
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_X, zeros_train_y = zeros_train_data
        zeros_train_X = zeros_train_X.iloc[2:]
        zeros_train_y = zeros_train_y[2:]
        selector.fit(
            X=zeros_train_X, y=zeros_train_y,
        )
        expected = zeros_train_X[["train_2", "train_5", "train_6"]]
        actual = selector.transform(zeros_train_X)
        assert_frame_equal(actual, expected)

    def test_alpha_full_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.8,
        )
        train_X, train_y = train_data
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[["train_1", "train_2", "train_3", "train_5", "train_6"]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_alpha_subset_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.8,
        )
        train_X, train_y = train_data
        train_X = train_X.iloc[1:] # subsetting train data
        train_y = train_y[1:]
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[["train_1", "train_2", "train_4", "train_5", "train_6"]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

class TestSelectPairedNullScoreOutlier:
    @fixture
    def score_func(self):
        def func(x, y):
            return x[0] + y[0]
        return func

    @fixture
    def null_data(self):
        data = DataFrame(
            index=[
                "sample_1",
                "sample_2",
                "sample_3",
                "sample_4",
            ],
            columns=[
                "col_1", "col_2", "col_3", "col_4", "col_5", "null_y",
            ],
            data=[
                [5, 1, 2, 3, 4,  1],
                [3, 4, 2, 1, 5,  0],
                [2, 4, 3, 1, 5, -1],
                [1, 2, 4, 3, 5,  0],
            ],
        )
        null_y = data.pop("null_y").to_numpy()
        # train_samples scores
        #           all   6, 2, 3, 4, 5
        #         1,2,3   6, 2, 3, 4, 5
        #           2,3   3, 4, 2, 1, 5
        #             3   1, 3, 2, 0, 4
        return (data, null_y)

    @fixture
    def train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "col_1", "col_2", "col_3", "col_4", "col_5", "train_y",
            ],
            data=[
                [1, 2, 5, 4, 3,   0],
                [3, 4, 5, 1, 2,  -1],
                [2, 5, 3, 1, 4,   1],
                [2, 5, 1, 3, 4,  -1],
            ],
        )
        train_y = data.pop("train_y").to_numpy()
        # train_samples    scores           delta_scores
        #           all    1, 2, 5, 4, 3    -5,  0,  2,  0, -2
        #       1, 2, 3    2, 3, 4, 0, 1    -4,  1,  1, -4, -4
        #          2, 3    3, 6, 4, 2, 5     0,  2,  2,  1,  0
        #             3    1, 4, 0, 2, 3     0,  1, -2,  2, -1

        # alpha train_samples lower_quantile upper_quantile  selected_columns
        #  0.25           all          -4.85            1.9     col_1, col_3
        #  0.25         1,2,3          -4.              1.      None
        #  0.25           2,3           0.              2.      None
        #  0.25             3          -1.95            1.95    col_3, col_4
        #  0.8            all          -1.04            0.      col_1, col_3, col_5
        #  0.8          1,2,3          -4.             -1.4     col_2, col_3
        #  0.8            2,3           0.48            1.52    col_1, col_2, col_3, col_5
        #  0.8              3          -0.52            0.52    col_2, col_3, col_4, col_5
        return (data, train_y)

    @fixture
    def zeros_train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "col_1", "col_2", "col_3", "col_4", "col_5", "train_y",
            ],
            data=[
                [0, 0, 0, 0, 0, -100],
                [0, 0, 0, 0, 0, -100],
                [0, 0, 0, 0, 0, -100],
                [0, 0, 0, 0, 0, -100],
            ],
        )
        train_y = data.pop("train_y").to_numpy()
        # would always all get selected
        return (data, train_y)

    def test_full_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectPairedNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
        )
        train_X, train_y = train_data
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[["col_1", "col_3"]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_subset_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectPairedNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
        )
        train_X, train_y = train_data
        train_X = train_X.iloc[1:] # subsetting train data
        train_y = train_y[1:]
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[[]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_full_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        null_X, null_y = null_data
        train_X, train_y = train_data
        selector = SelectPairedNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            train_X=train_X,
            train_y=train_y,
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_X, zeros_train_y = zeros_train_data
        selector.fit(
            X=zeros_train_X, y=zeros_train_y,
        )
        expected = zeros_train_X[["col_1", "col_3"]]
        actual = selector.transform(zeros_train_X)
        assert_frame_equal(actual, expected)

    def test_subset_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        null_X, null_y = null_data
        train_X, train_y = train_data
        selector = SelectPairedNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            train_X=train_X,
            train_y=train_y,
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_X, zeros_train_y = zeros_train_data
        zeros_train_X = zeros_train_X.iloc[3:]
        zeros_train_y = zeros_train_y[3:]
        selector.fit(
            X=zeros_train_X, y=zeros_train_y,
        )
        expected = zeros_train_X[["col_3", "col_4"]]
        actual = selector.transform(zeros_train_X)
        assert_frame_equal(actual, expected)

    def test_alpha_full_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectPairedNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.8,
        )
        train_X, train_y = train_data
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[["col_1", "col_3", "col_5"]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_alpha_subset_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = SelectPairedNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.8,
        )
        train_X, train_y = train_data
        train_X = train_X.iloc[1:] # subsetting train data
        train_y = train_y[1:]
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X[["col_2", "col_3"]]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

class TestAggregateNullScoreOutlier:
    @fixture
    def score_func(self):
        def func(x, y):
            return x[0] + y[0]
        return func

    @fixture
    def null_data(self):
        data = DataFrame(
            index=[
                "sample_1",
                "sample_2",
                "sample_3",
                "sample_4",
            ],
            columns=[
                "null_1", "null_2", "null_3", "null_4", "null_5", "null_y",
            ],
            data=[
                [5, 1, 2, 3, 4,  1],
                [3, 4, 2, 1, 5,  0],
                [2, 4, 3, 1, 5, -1],
                [1, 2, 4, 3, 5,  0],
            ],
        )
        null_y = data.pop("null_y").to_numpy()
        # alpha first_train_row lower_quantile upper_quantile
        #  0.25        sample_1           2.05           5.95
        #  0.25        sample_2           1.05           4.95
        #  0.25        sample_3           0.05           3.95
        #  0.25        sample_4           1.05           4.95
        #  0.8         sample_1           3.48           4.52
        #  0.8         sample_2           2.48           3.52
        #  0.8         sample_3           1.48           2.52
        #  0.8         sample_4           2.48           3.52
        return (data, null_y)

    @fixture
    def train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_y",
            ],
            data=[
                [1, 2, 5, 4, 3, 1,   0],
                [3, 4, 5, 1, 2, 3,  -1],
                [2, 5, 3, 1, 4, 5,   1],
                [2, 5, 1, 3, 4, 4,  -1],
            ],
        )
        train_y = data.pop("train_y").to_numpy()
        # alpha train_samples lower_quantile upper_quantile selected_features
        #  0.25           all           2.05           5.95   train_1, train_2, train_6
        #  0.25       1, 2, 3           2.05           5.95   train_1, train_4, train_5, train_6
        #  0.25          2, 3           1.05           4.95   train_2, train_5, train_6
        #  0.25             3           0.05           3.95   train_2, train_3
        #  0.8            all           3.48           4.52   train_1, train_2, train_3, train_5, train_6
        #  0.8        1, 2, 3           3.48           4.52   train_1, train_2, train_4, train_5, train_6
        #  0.8           2, 3           2.48           3.52   train_2, train_3, train_4, train_5, train_6
        #  0.8              3           1.48           2.52   train_1, train_2, train_3, train_5, train_6
        return (data, train_y)

    @fixture
    def zeros_train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_y",
            ],
            data=[
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
            ],
        )
        train_y = data.pop("train_y").to_numpy()
        # would always all get selected
        return (data, train_y)

    def test_full_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = AggregateNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
        )
        train_X, train_y = train_data
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = (
            train_X[["train_1", "train_2", "train_6"]]
            .sum(axis=1)
            .to_frame()
        )
        expected.columns = ["train_1;train_2;train_6"]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_flatten_columns(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        null_X.columns = MultiIndex.from_arrays(
            names=[f"level_{j}" for j in range(2)],
            arrays=[
                [f"level_{j}_col_{i}" for i in range(null_X.shape[1])]
                for j in range(2)
            ]
        )
        selector = AggregateNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
            flatten_columns=True,
        )
        train_X, train_y = train_data
        train_X.columns = MultiIndex.from_arrays(
            names=[f"level_{j}" for j in range(2)],
            arrays=[
                [f"level_{j}_col_{i}" for i in range(train_X.shape[1])]
                for j in range(2)
            ]
        )
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = train_X.copy()
        expected.columns = expected.columns.to_flat_index().map(str)
        expected.columns.name = str(("level_0", "level_1"))
        expected = (
            expected[[
                str(("level_0_col_0", "level_1_col_0")),
                str(("level_0_col_1", "level_1_col_1")),
                str(("level_0_col_5", "level_1_col_5")),
            ]]
            .sum(axis=1)
            .to_frame()
        )
        expected.columns = [";".join([
            str(("level_0_col_0", "level_1_col_0")),
            str(("level_0_col_1", "level_1_col_1")),
            str(("level_0_col_5", "level_1_col_5")),
        ])]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_subset_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = AggregateNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.25,
        )
        train_X, train_y = train_data
        train_X = train_X.iloc[1:] # subsetting train data
        train_y = train_y[1:]
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = (
            train_X[["train_1", "train_4", "train_5", "train_6"]]
            .sum(axis=1)
            .to_frame()
        )
        expected.columns = ["train_1;train_4;train_5;train_6"]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_full_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        null_X, null_y = null_data
        train_X, train_y = train_data
        selector = AggregateNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            train_X=train_X,
            train_y=train_y,
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_X, zeros_train_y = zeros_train_data
        selector.fit(
            X=zeros_train_X, y=zeros_train_y,
        )
        expected = (
            zeros_train_X[["train_1", "train_2", "train_6"]]
            .sum(axis=1)
            .to_frame()
        )
        expected.columns = ["train_1;train_2;train_6"]
        actual = selector.transform(zeros_train_X)
        assert_frame_equal(actual, expected)

    def test_subset_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        null_X, null_y = null_data
        train_X, train_y = train_data
        selector = AggregateNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            train_X=train_X,
            train_y=train_y,
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_X, zeros_train_y = zeros_train_data
        zeros_train_X = zeros_train_X.iloc[2:]
        zeros_train_y = zeros_train_y[2:]
        selector.fit(
            X=zeros_train_X, y=zeros_train_y,
        )
        expected = (
            zeros_train_X[["train_2", "train_5", "train_6"]]
            .sum(axis=1)
            .to_frame()
        )
        expected.columns = ["train_2;train_5;train_6"]
        actual = selector.transform(zeros_train_X)
        assert_frame_equal(actual, expected)

    def test_alpha_full_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = AggregateNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.8,
        )
        train_X, train_y = train_data
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = (
            train_X[["train_1", "train_2", "train_3", "train_5", "train_6"]]
            .sum(axis=1)
            .to_frame()
        )
        expected.columns = ["train_1;train_2;train_3;train_5;train_6"]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)

    def test_alpha_subset_train(self, null_data, train_data, score_func):
        null_X, null_y = null_data
        selector = AggregateNullScoreOutlier(
            null_X=null_X,
            null_y=null_y,
            score_func=score_func,
            alpha=0.8,
        )
        train_X, train_y = train_data
        train_X = train_X.iloc[1:] # subsetting train data
        train_y = train_y[1:]
        selector.fit(
            X=train_X, y=train_y,
        )
        expected = (
            train_X[["train_1", "train_2", "train_4", "train_5", "train_6"]]
            .sum(axis=1)
            .to_frame()
        )
        expected.columns = ["train_1;train_2;train_4;train_5;train_6"]
        actual = selector.transform(train_X)
        assert_frame_equal(actual, expected)