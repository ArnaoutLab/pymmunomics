import warnings

from numpy import isclose, nan
from pandas import DataFrame, isna, Index, MultiIndex, read_csv, read_excel, Series
from pandas.testing import assert_frame_equal
from pytest import fixture, raises, warns

from pymmunomics.helper.exception import (
    AmbiguousValuesWarning,
    DivergingValuesWarning,
    InvalidArgumentError,
)
from pymmunomics.helper.pandas_helpers import (
    agg_first_safely,
    apply_zipped,
    assert_groups_equal,
    concat_partial_groupby_apply,
    concat_pivot_pipe_melt,
    concat_weighted_value_counts,
    pipe_assign_from_func,
    read_mapping,
    read_combine_mappings,
    remove_duplicated_rows,
    squash_column_safely,
    weighted_mean,
    weighted_variance,
    weighted_skewness,
)

class TestAggFirstSafely:
    def test_drop_nans_nonambiguous(self):
        series = Series([1, nan, 1])
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=AmbiguousValuesWarning)
            actual_result = agg_first_safely(
                series=series,
                dropna=True,
            )
        assert actual_result == 1

    def test_drop_nans_ambiguous(self):
        series = Series([1, nan, 2, 1])
        with warns(AmbiguousValuesWarning):
            actual_result = agg_first_safely(
                series=series,
                dropna=True,
            )
        assert actual_result == 1

    def test_keep_nans_nonambiguous(self):
        series = Series(["foo", "foo", "foo"])
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=AmbiguousValuesWarning)
            actual_result = agg_first_safely(
                series=series,
                dropna=False,
            )
        assert actual_result == "foo"

    def test_keep_nans_ambiguous(self):
        series = Series(["foo", nan, "foo", "foo"])
        with warns(AmbiguousValuesWarning):
            actual_result = agg_first_safely(
                series=series,
                dropna=False,
            )
        assert actual_result == "foo"

    def test_nan_result_nonambiguous(self):
        series = Series([nan, nan, nan])
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=AmbiguousValuesWarning)
            actual_result = agg_first_safely(
                series=series,
                dropna=False,
            )
        assert isna(actual_result)

    def test_nan_result_ambiguous(self):
        series = Series([nan, "foo"])
        with warns(AmbiguousValuesWarning):
            actual_result = agg_first_safely(
                series=series,
                dropna=False,
            )
        assert isna(actual_result)

class TestApplyZipped:
    def fake_func(*args, **kwargs):
        return (args, kwargs)

    @fixture
    def data_frame(self):
        return DataFrame(
            columns=["a", "b", "c", "d"],
            data=[
                [1, 10, "x", "foo"],
                [2, 20, "x", "foo"],
                [3, 30, "y", "foo"],                
            ]
        )

    def test_standard_input(self, data_frame):
        keys = ["a", "c"]
        func = TestApplyZipped.fake_func
        expected_result = [
            ((1, "x"), {}),
            ((2, "x"), {}),
            ((3, "y"), {}),
        ]
        actual_result = apply_zipped(
            data_frame=data_frame,
            keys=keys,
            func=func,
        )
        assert actual_result == expected_result

    def test_packed(self, data_frame):
        keys = ["a", "c"]
        func = TestApplyZipped.fake_func
        expected_result = [
            (((1, "x"),), {}),
            (((2, "x"),), {}),
            (((3, "y"),), {}),
        ]
        actual_result = apply_zipped(
            data_frame=data_frame,
            keys=keys,
            func=func,
            unpack=False,
        )
        assert actual_result == expected_result

    def test_single_key(self, data_frame):
        keys = ["c"]
        func = TestApplyZipped.fake_func
        expected_result = [
            (("x",), {}),
            (("x",), {}),
            (("y",), {}),
        ]
        actual_result = apply_zipped(
            data_frame=data_frame,
            keys=keys,
            func=func,
        )
        assert actual_result == expected_result

    def test_single_key_packed(self, data_frame):
        keys = ["c"]
        func = TestApplyZipped.fake_func
        expected_result = [
            ((("x",),), {}),
            ((("x",),), {}),
            ((("y",),), {}),
        ]
        actual_result = apply_zipped(
            data_frame=data_frame,
            keys=keys,
            func=func,
            unpack=False,
        )
        assert actual_result == expected_result

    def test_args_kwargs(self, data_frame):
        keys = ["a", "c"]
        func = TestApplyZipped.fake_func
        expected_result = [
            ((1, "x", "foo", 0), {"zip": "zap"}),
            ((2, "x", "foo", 0), {"zip": "zap"}),
            ((3, "y", "foo", 0), {"zip": "zap"}),
        ]
        actual_result = apply_zipped(
            data_frame,
            keys,
            func,
            "foo",
            0,
            zip="zap",
        )
        assert actual_result == expected_result

class TestAssertGroupsEqual:
    def test_nopipe(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        with raises(AssertionError):
            assert_groups_equal(
                data_frame=data_frame,
                groupby_kwargs={"by": ["g1", "g2"]},
            )

    def test_equal(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        assert_groups_equal(
            data_frame=data_frame,
            groupby_kwargs={"by": ["g1", "g2"]},
            group_pipe=lambda df: df[["val1"]].reset_index(drop=True),
        )

    def test_different(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        with raises(AssertionError):
            assert_groups_equal(
                data_frame=data_frame,
                groupby_kwargs={"by": ["g1", "g2"]},
                group_pipe=lambda df: df[["val2"]].reset_index(drop=True),
            )

    def test_float_close_assert_frame_equal_kwargs(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1 + 1e-3, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        assert_groups_equal(
            data_frame=data_frame,
            groupby_kwargs={"by": ["g1", "g2"]},
            group_pipe=lambda df: df[["val1"]].reset_index(drop=True),
            assert_frame_equal_kwargs={"check_exact": False, "rtol": 0, "atol": 1e-2},
        )

    def test_float_differ_assert_frame_equal_kwargs(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1 + 1e-3, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        with raises(AssertionError):
            assert_groups_equal(
                data_frame=data_frame,
                groupby_kwargs={"by": ["g1", "g2"]},
                group_pipe=lambda df: (df[["val1"]].reset_index(drop=True)),
                assert_frame_equal_kwargs={"check_exact": False, "rtol": 0, "atol": 1e-4},
            )


class TestConcatPartialGroupbyApply:
    def test_some_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ["a", "pooled"],
                    ["b", "pooled"],
                    ["pooled", "a"],
                    ["pooled", "b"],
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2 * 7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 3 * 5 * 13],
                [7 * 11],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[["group2"], ["group1"]],
        )
        assert_frame_equal(actual, expected)

    def test_all_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("pooled", "pooled"),
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2 * 3 * 5 * 7 * 11 * 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[["group1", "group2"]],
        )
        assert_frame_equal(actual, expected)

    def test_none_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2],
                [7 * 11],
                [3 * 5 * 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
        )
        assert_frame_equal(actual, expected)

    def test_mixture_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "pooled"),
                    ("b", "pooled"),
                    ("pooled", "pooled"),
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2],
                [7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 3 * 5 * 7 * 11 * 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[[], ["group2"], ["group1", "group2"]],
        )
        assert_frame_equal(actual, expected)

    def test_multiple_func(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "pooled"),
                    ("b", "pooled"),
                    ("pooled", "pooled"),
                ],
            ),
            columns=["val", "val"],
            data=[
                [1 * 2, 1 + 2],
                [7 * 11, 7 + 11],
                [3 * 5 * 13, 3 + 5 + 13],
                [1 * 2 * 7 * 11, 1 + 2 + 7 + 11],
                [3 * 5 * 13, 3 + 5 + 13],
                [1 * 2 * 3 * 5 * 7 * 11 * 13, 1 + 2 + 3 + 5 + 7 + 11 + 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=[
                lambda df: df[["val"]].prod(),
                lambda df: df[["val"]].sum(),
            ],
            by=["group1", "group2"],
            pooled=[[], ["group2"], ["group1", "group2"]],
        )
        assert_frame_equal(actual, expected)

    def test_multiple_func_with_keys(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "pooled"),
                    ("b", "pooled"),
                    ("pooled", "pooled"),
                ],
            ),
            columns=MultiIndex.from_tuples(
                [("prod", "val"), ("sum", "val")],
            ),
            data=[
                [1 * 2, 1 + 2],
                [7 * 11, 7 + 11],
                [3 * 5 * 13, 3 + 5 + 13],
                [1 * 2 * 7 * 11, 1 + 2 + 7 + 11],
                [3 * 5 * 13, 3 + 5 + 13],
                [1 * 2 * 3 * 5 * 7 * 11 * 13, 1 + 2 + 3 + 5 + 7 + 11 + 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=[
                lambda df: df[["val"]].prod(),
                lambda df: df[["val"]].sum(),
            ],
            by=["group1", "group2"],
            pooled=[[], ["group2"], ["group1", "group2"]],
            func_keys=["prod", "sum"],
        )
        assert_frame_equal(actual, expected)

    def test_single_func_with_keys_and_names(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "pooled"),
                    ("b", "pooled"),
                    ("pooled", "pooled"),
                ],
            ),
            columns=MultiIndex.from_tuples(
                [("prod", "val")],
                names=["func", "col"],
            ),
            data=[
                [1 * 2],
                [7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 3 * 5 * 7 * 11 * 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[[], ["group2"], ["group1", "group2"]],
            func_keys=["prod"],
            col_names=["func", "col"],
        )
        assert_frame_equal(actual, expected)

    def test_multiple_func_with_keys_and_names(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "pooled"),
                    ("b", "pooled"),
                    ("pooled", "pooled"),
                ],
            ),
            columns=MultiIndex.from_tuples(
                [("prod", "val"), ("sum", "val")],
                names=["func", "col"],
            ),
            data=[
                [1 * 2,                       1 + 2],
                [7 * 11,                      7 + 11],
                [3 * 5 * 13,                  3 + 5 + 13],
                [1 * 2 * 7 * 11,              1 + 2 + 7 + 11],
                [3 * 5 * 13,                  3 + 5 + 13],
                [1 * 2 * 3 * 5 * 7 * 11 * 13, 1 + 2 + 3 + 5 + 7 + 11 + 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=[
                lambda df: df[["val"]].prod(),
                lambda df: df[["val"]].sum(),
            ],
            by=["group1", "group2"],
            pooled=[[], ["group2"], ["group1", "group2"]],
            func_keys=["prod", "sum"],
            col_names=["func", "col"],
        )
        assert_frame_equal(actual, expected)


class TestConcatPivotPipeMelt:
    def test_multiple_pivot_values(self):
        # pivotted:
        # val1                          val2
        #     col a   b   c                 col a   b   c
        #  idx                           idx
        #   a     1   2   3               a     1   nan 3
        #   b     4   5   nan             b     4   nan nan
        data_frame = DataFrame(
            index=Index(
                name="idx",
                data=[
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                ],
            ),
            columns=["col", "val1", "val2"],
            data=[
                ["a", 1, 1],
                ["b", 2, nan],
                ["c", 3, 3],
                ["a", 4, 4],
                ["b", 5, nan],
            ],
        )
        expected = DataFrame(
            index=Index(
                name="col",
                data=[
                    "a",
                    "a",
                    "b",
                    "b",
                    "c",
                    "c",
                ],
            ),
            columns=["val1", "val2"],
            data=[
                [1., 1.],
                [4., 4.],
                [2., 0.],
                [5., 0.],
                [3., 3.],
                [0., 0.],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val1", "val2"],
            columns="col",
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_multiple_pivot_columns(self):
        data_frame = DataFrame(
            index=Index(
                name="idx",
                data=[
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    "b",
                    "b",
                ],
            ),
            columns=["col1", "col2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "b", 2],
                ["b", "b", 4],
                ["c", "a", 5],
                ["c", "b", 6],
                ["b", "a", 9],
                ["b", "b", 10],
                ["c", "a", 11],
                ["c", "b", 12],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["col1", "col2"],
                tuples=[
                    ("a", "a"),
                    ("a", "a"),
                    ("a", "b"),
                    ("a", "b"),
                    ("b", "a"),
                    ("b", "a"),
                    ("b", "b"),
                    ("b", "b"),
                    ("c", "a"),
                    ("c", "a"),
                    ("c", "b"),
                    ("c", "b"),
                ],
            ),
            columns=["val"],
            data=[
                [1.0],
                [0.0],
                [2.0],
                [0.0],
                [0.0],
                [9.0],
                [4.0],
                [10.0],
                [5.0],
                [11.0],
                [6.0],
                [12.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val"],
            columns=["col1", "col2"],
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_single_pivot_index(self):
        # pivotted:
        # val
        #     col a   b   c
        #  idx
        #   a     1   2   3
        #   b     4   5   nan
        data_frame = DataFrame(
            columns=["idx", "col", "val"],
            data=[
                ["a", "a", 1],
                ["a", "b", 2],
                ["a", "c", 3],
                ["b", "a", 4],
                ["b", "b", 5],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["idx", "col"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("a", "c"),
                    ("b", "a"),
                    ("b", "b"),
                    ("b", "c"),
                ],
            ),
            columns=["val"],
            data=[
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [0.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val"],
            columns="col",
            index="idx",
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_multiple_pivot_index(self):
        # pivotted:
        #     col      a   b
        #  idx1 idx2
        #   a    a     1   2
        #   a    b     nan 4
        #   a    c     5   6
        #   b    b     9   10
        #   b    c     11  12
        data_frame = DataFrame(
            columns=["idx1", "idx2", "col", "val"],
            data=[
                ["a", "a", "a", 1],
                ["a", "a", "b", 2],
                ["a", "b", "b", 4],
                ["a", "c", "a", 5],
                ["a", "c", "b", 6],
                ["b", "b", "a", 9],
                ["b", "b", "b", 10],
                ["b", "c", "a", 11],
                ["b", "c", "b", 12],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["idx1", "idx2", "col"],
                tuples=[
                    ('a', 'a', 'a'),
                    ('a', 'a', 'b'),
                    ('a', 'b', 'a'),
                    ('a', 'b', 'b'),
                    ('a', 'c', 'a'),
                    ('a', 'c', 'b'),
                    ('b', 'b', 'a'),
                    ('b', 'b', 'b'),
                    ('b', 'c', 'a'),
                    ('b', 'c', 'b'),
                ],
            ),
            columns=["val"],
            data=[
                [ 1.0],
                [ 2.0],
                [ 0.0],
                [ 4.0],
                [ 5.0],
                [ 6.0],
                [ 9.0],
                [10.0],
                [11.0],
                [12.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val"],
            columns="col",
            index=["idx1", "idx2"],
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_mixture(self):
        data_frame = DataFrame(
            columns=["idx", "col", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "b", 2, nan],
                ["a", "c", 3, 3],
                ["b", "a", 4, 4],
                ["b", "b", 5, nan],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["idx", "col"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("a", "c"),
                    ("b", "a"),
                    ("b", "b"),
                    ("b", "c"),
                ],
            ),
            columns=["val1", "val2"],
            data=[
                [1.0, 1.0],
                [2.0, 0.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 0.0],
                [0.0, 0.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val1", "val2"],
            columns="col",
            index=["idx"],
            value=0,
        )
        assert_frame_equal(actual, expected)

class TestConcatWeightedValueCounts:
    def test_single_column(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                ],
            ),
            columns=["count"],
            data=[
                [11],
                [1100],
                [10000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1"],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_multiple_columns(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                    ("column_2", "x"),
                    ("column_2", "y"),
                ],
            ),
            columns=["count"],
            data=[
                [11],
                [1100],
                [10000],
                [111],
                [11000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1", "column_2"],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_compound_column(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["count"],
            data=[
                [11],
                [100],
                [1000],
                [10000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=[["column_1", "column_2"]],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_mixed_columns(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_2", "x"),
                    ("column_2", "y"),
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["count"],
            data=[
                [111],
                [11000],
                [11],
                [100],
                [1000],
                [10000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_2", ["column_1", "column_2"]],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_single_column_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                ],
            ),
            columns=["frequency"],
            data=[
                [11 / 11111],
                [1100 / 11111],
                [10000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1"],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)

    def test_multiple_columns_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                    ("column_2", "x"),
                    ("column_2", "y"),
                ],
            ),
            columns=["frequency"],
            data=[
                [11 / 11111],
                [1100 / 11111],
                [10000 / 11111],
                [111 / 11111],
                [11000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1", "column_2"],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)

    def test_compound_column_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["frequency"],
            data=[
                [11 / 11111],
                [100 / 11111],
                [1000 / 11111],
                [10000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=[["column_1", "column_2"]],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)

    def test_mixed_columns_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_2", "x"),
                    ("column_2", "y"),
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["frequency"],
            data=[
                [111 / 11111],
                [11000 / 11111],
                [11 / 11111],
                [100 / 11111],
                [1000 / 11111],
                [10000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_2", ["column_1", "column_2"]],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)

class TestPipeAssignFromFunc:
    def fake_func_single_column(data, *args, **kwargs):
        if "add" in kwargs:
            add = kwargs["add"]
        else:
            add = 1
        return (data + add).sum(axis=1).to_numpy()

    def fake_func_multiple_columns(data, *args, **kwargs):
        if "add" in kwargs:
            add = kwargs["add"]
        else:
            add = 1
        return (data + add).to_numpy()

    @fixture
    def data_frame(self):
        return DataFrame(
            columns=["a", "b"],
            data=[
                [1, 10],
                [2, 20],
                [3, 30],
            ]
        )

    def test_single_column(self, data_frame):
        names = "c"
        pipe_func = TestPipeAssignFromFunc.fake_func_single_column

        expected_result = DataFrame(
            columns=["a", "b", "c"],
            data=[
                [1, 10, 13],
                [2, 20, 24],
                [3, 30, 35],
            ]
        )
        actual_result = pipe_assign_from_func(
            data_frame=data_frame,
            names=names,
            pipe_func=pipe_func,
        )
        assert_frame_equal(actual_result, expected_result)
        assert not (actual_result is data_frame)

    def test_multiple_columns(self, data_frame):
        names = ["c", "d"]
        pipe_func = TestPipeAssignFromFunc.fake_func_multiple_columns

        expected_result = DataFrame(
            columns=["a", "b", "c", "d"],
            data=[
                [1, 10, 2, 11],
                [2, 20, 3, 21],
                [3, 30, 4, 31],
            ]
        )
        actual_result = pipe_assign_from_func(
            data_frame=data_frame,
            names=names,
            pipe_func=pipe_func,
        )
        assert_frame_equal(actual_result, expected_result)
        assert not (actual_result is data_frame)

    def test_inplace(self, data_frame):
        data_frame_ = data_frame.copy()
        names = "c"
        pipe_func = TestPipeAssignFromFunc.fake_func_single_column

        expected_result = DataFrame(
            columns=["a", "b", "c"],
            data=[
                [1, 10, 13],
                [2, 20, 24],
                [3, 30, 35],
            ]
        )
        actual_result = pipe_assign_from_func(
            data_frame=data_frame_,
            names=names,
            pipe_func=pipe_func,
            inplace=True,
        )
        assert_frame_equal(actual_result, expected_result)
        assert actual_result is data_frame_

    def test_kwargs(self, data_frame):
        names = "c"
        pipe_func = TestPipeAssignFromFunc.fake_func_single_column

        expected_result = DataFrame(
            columns=["a", "b", "c"],
            data=[
                [1, 10, 21],
                [2, 20, 32],
                [3, 30, 43],
            ]
        )
        actual_result = pipe_assign_from_func(
            data_frame=data_frame,
            names=names,
            pipe_func=pipe_func,
            add=5,
        )
        assert_frame_equal(actual_result, expected_result)
        assert not (actual_result is data_frame)

class TestReadMapping:
    @fixture
    def table(self):
        return DataFrame(
            columns=["col1", "col2", "col3", "col4"],
            data=[
                [1, 10, "foo", "x"],
                [nan, 20, "bar", "y"],
                [3, 30, "foo", "z"],
            ],
        )

    def test_single_key(self, tmp_path, table):
        filepath = f"{tmp_path}/file.csv"
        key = "col2"
        value = "col4"

        table.to_csv(filepath, index=False)

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
        }
        actual_mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
        )

        assert actual_mapping == expected_mapping

    def test_single_key_list(self, tmp_path, table):
        filepath = f"{tmp_path}/file.csv"
        key = ["col2"]
        value = "col4"

        table.to_csv(filepath, index=False)

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
        }
        actual_mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
        )

        assert actual_mapping == expected_mapping

    def test_multiple_keys(self, tmp_path, table):
        filepath = f"{tmp_path}/file.csv"
        key = ["col3", "col2"]
        value = "col4"

        table.to_csv(filepath, index=False)

        expected_mapping = {
            ("foo", 10): "x",
            ("bar", 20): "y",
            ("foo", 30): "z",
        }
        actual_mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
        )

        assert actual_mapping == expected_mapping

    def test_drop_nan_key(self, tmp_path, table):
        filepath = f"{tmp_path}/file.csv"
        key = ["col1", "col2"]
        value = "col4"

        table.to_csv(filepath, index=False)

        expected_mapping = {
            (1, 10): "x",
            (3, 30): "z",
        }
        actual_mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
        )

        assert actual_mapping == expected_mapping

    def test_drop_nan_value(self, tmp_path, table):
        filepath = f"{tmp_path}/file.csv"
        key = "col2"
        value = "col1"

        table.to_csv(filepath, index=False)

        expected_mapping = {
            10: 1,
            30: 3,
        }
        actual_mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
        )

        assert actual_mapping == expected_mapping

    def test_read_func(self, tmp_path, table):
        filepath = f"{tmp_path}/file.xlsx"
        key = ["col2"]
        value = "col4"

        table.to_excel(filepath, index=False)

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
        }
        actual_mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
            read_func=read_excel,
        )

        assert actual_mapping == expected_mapping

    def test_read_kwargs(self, tmp_path, table):
        filepath = f"{tmp_path}/file.csv"
        key = ["col2"]
        value = "col4"

        table.to_csv(filepath, index=False, sep="\t")

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
        }
        actual_mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
            read_kwargs={"sep": "\t"},
        )

        assert actual_mapping == expected_mapping

class TestReadCombineMappings:
    @fixture
    def tables(self):
        return [
            DataFrame(
                columns=["col1", "col2", "col3", "col4"],
                data=[
                    [1, 10, "foo", "x"],
                    [nan, 20, "bar", "y"],
                    [3, 30, "foo", "z"],
                ],
            ),
            DataFrame(
                columns=["col1", "col2", "col3", "col4"],
                data=[
                    [4, 40, "bar", "x"],
                    [5, 50, "bar", "y"],
                    [nan, 60, "bar", "z"],
                ],
            ),
        ]

    def test_single_key(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = ["col2", "col2"]
        values = ["col4", "col4"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
            40: "x",
            50: "y",
            60: "z",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_single_key_list(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = [["col2"], ["col2"]]
        values = ["col4", "col4"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
            40: "x",
            50: "y",
            60: "z",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_multiple_keys(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = [["col3", "col2"], ["col3", "col2"]]
        values = ["col4", "col4"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            ("foo", 10): "x",
            ("bar", 20): "y",
            ("foo", 30): "z",
            ("bar", 40): "x",
            ("bar", 50): "y",
            ("bar", 60): "z",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_drop_nan_key(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = [["col1", "col2"], ["col1", "col2"]]
        values = ["col4", "col4"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            (1, 10): "x",
            (3, 30): "z",
            (4, 40): "x",
            (5, 50): "y",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_drop_nan_value(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = ["col2", "col2"]
        values = ["col1", "col1"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            10: 1,
            30: 3,
            40: 4,
            50: 5,
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_read_funcs(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file0.xlsx",
            f"{tmp_path}/file1.csv",
        ]
        keys = ["col2", "col2"]
        values = ["col4", "col4"]
        read_funcs = [read_excel, read_csv]

        tables[0].to_excel(filepaths[0], index=False)
        tables[1].to_csv(filepaths[1], index=False)

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
            40: "x",
            50: "y",
            60: "z",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=read_funcs,
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_read_kwargs(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = ["col2", "col2"]
        values = ["col4", "col4"]
        read_kwargs = [{"sep": "\t"}, {"comment": "#"}]

        tables[0].to_csv(filepaths[0], sep="\t", index=False)
        with open(filepaths[1], "w") as file:
            file.write("# comment line 1\n")
            file.write("# comment line 2\n")
            tables[1].to_csv(file, index=False)

        expected_mapping = {
            10: "x",
            20: "y",
            30: "z",
            40: "x",
            50: "y",
            60: "z",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=read_kwargs,
        )

        assert actual_mapping == expected_mapping

    def test_mixed_keys_values(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = [["col3", "col2"], ["col1"]]
        values = ["col4", "col2"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            ("foo", 10): "x",
            ("bar", 20): "y",
            ("foo", 30): "z",
            4: 40,
            5: 50,
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_single_table(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = [["col3", "col2"]]
        values = ["col4"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            ("foo", 10): "x",
            ("bar", 20): "y",
            ("foo", 30): "z",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths[:1],
            keys=keys,
            values=values,
            read_funcs=[read_csv],
            read_kwargs=[{}],
        )

        assert actual_mapping == expected_mapping

    def test_transiftives_duplicates(self, tmp_path):
        tables = [
            DataFrame(
                columns=["col1", "col2", "col3", "col4"],
                data=[
                    [1, 10, "a", "x"],
                    [2, 20, "b", "y"],
                    [3, 30, "c", "z"],
                ],
            ),
            DataFrame(
                columns=["col1", "col2", "col3", "col4"],
                data=[
                    [4, 40, "a", "y"],
                    [5, 50, "z", "x"],
                    [6, 60, "y", "z"],
                ],
            ),
        ]
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = [["col3"], ["col3"]]
        values = ["col4", "col4"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        expected_mapping = {
            "b": "z",
            "c": "x",
            "a": "z",
            "z": "x",
            "y": "z",
        }
        actual_mapping = read_combine_mappings(
            filepaths=filepaths,
            keys=keys,
            values=values,
            read_funcs=[read_csv, read_csv],
            read_kwargs=[{}, {}],
        )

        assert actual_mapping == expected_mapping

    def test_invalid_argument(self, tmp_path, tables):
        filepaths = [
            f"{tmp_path}/file{i}.csv"
            for i in range(len(tables))
        ]
        keys = ["col2", "col2"]
        values = ["col4", "col4"]

        for table, filepath in zip(tables, filepaths):
            table.to_csv(filepath, index=False)

        with raises(InvalidArgumentError):
            read_combine_mappings(
                filepaths=filepaths[:1],
                keys=keys,
                values=values,
                read_funcs=[read_csv, read_csv],
                read_kwargs=[{}, {}],
            )

        with raises(InvalidArgumentError):
            read_combine_mappings(
                filepaths=filepaths,
                keys=keys[:1],
                values=values,
                read_funcs=[read_csv, read_csv],
                read_kwargs=[{}, {}],
            )

        with raises(InvalidArgumentError):
            read_combine_mappings(
                filepaths=filepaths,
                keys=keys,
                values=values[:1],
                read_funcs=[read_csv, read_csv],
                read_kwargs=[{}, {}],
            )

        with raises(InvalidArgumentError):
            read_combine_mappings(
                filepaths=filepaths,
                keys=keys,
                values=values,
                read_funcs=[read_csv],
                read_kwargs=[{}, {}],
            )

        with raises(InvalidArgumentError):
            read_combine_mappings(
                filepaths=filepaths,
                keys=keys,
                values=values,
                read_funcs=[read_csv, read_csv],
                read_kwargs=[{}],
            )

class TestRemoveDuplicatedRows:
    @fixture
    def data_frame(self):
        return DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )

    def test_single_identity_value_nondiverging(self, data_frame):
        identity_columns = ["id2"]
        value_columns = ["val1"]
        equal_nan = False

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,2],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,3,4,5,6,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=DivergingValuesWarning)
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)

    def test_single_identity_value_diverging(self, data_frame):
        identity_columns = ["id1"]
        value_columns = ["val1"]
        equal_nan = False

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,4],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["b", "a", "a", 1, 10, 100,  nan],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,2,3,5,6,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warns(DivergingValuesWarning):
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)

    def test_multiple_identity_single_value_nondiverging(self, data_frame):
        identity_columns = ["id1", "id2"]
        value_columns = ["val1"]
        equal_nan = False

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,2,4,6],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,3,5,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=DivergingValuesWarning)
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)

    def test_multiple_identity_multiple_value_diverging(self, data_frame):
        identity_columns = ["id1", "id2"]
        value_columns = ["val1", "val2"]
        equal_nan = False

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,2,4,6],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,3,5,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warns(DivergingValuesWarning):
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)

    def test_single_identity_equal_nan_value_nondiverging(self, data_frame):
        identity_columns = ["id2"]
        value_columns = ["val4"]
        equal_nan = True

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,2],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,3,4,5,6,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=DivergingValuesWarning)
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)

    def test_single_identity_equal_nan_value_diverging(self, data_frame):
        identity_columns = ["id1"]
        value_columns = ["val3"]
        equal_nan = True

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,4],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["b", "a", "a", 1, 10, 100,  nan],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,2,3,5,6,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warns(DivergingValuesWarning):
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)

    def test_multiple_identity_single_equal_nan_value_nondiverging(self, data_frame):
        identity_columns = ["id1", "id2"]
        value_columns = ["val4"]
        equal_nan = True

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,2],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,3,4,5,6,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=DivergingValuesWarning)
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )

    def test_multiple_identity_multiple_nan_value_diverging(self, data_frame):
        identity_columns = ["id1", "id2"]
        value_columns = ["val1", "val3"]
        equal_nan = True

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,2,4,6],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,3,5,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warns(DivergingValuesWarning):
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)

    def test_non_equal_nan_value_diverging(self, data_frame):
        identity_columns = ["id2"]
        value_columns = ["val3"]
        equal_nan = False

        expected_data_frame = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[0,2],
            data=[
                ["a", "a", "a", 1, 10, 100,  nan],
                ["a", "b", "a", 3, 30, 300, 2000],
            ],
        ).astype({"val3": float})
        expected_duplicated = DataFrame(
            columns=["id1", "id2", "id3", "val1", "val2", "val3", "val4"],
            index=[1,3,4,5,6,7],
            data=[
                ["a", "a", "b", 1, 20, nan,  nan],
                ["a", "b", "b", 3, 40, 300, 2000],
                ["b", "a", "a", 1, 10, 100,  nan],
                ["b", "a", "b", 1, 20, nan,  nan],
                ["b", "b", "a", 3, 30, 300, 2000],
                ["b", "b", "b", 3, 40, 300, 2000],
            ],
        )
        with warns(DivergingValuesWarning):
            actual_data_frame, actual_duplicated = remove_duplicated_rows(
                data_frame=data_frame,
                identity_columns=identity_columns,
                value_columns=value_columns,
                equal_nan=equal_nan,
            )
        assert_frame_equal(actual_data_frame, expected_data_frame)
        assert_frame_equal(actual_duplicated, expected_duplicated)


class TestSquashColumnSafely:
    def test_simple_input(self):
        column = Series(["foo", "foo", "foo"])
        preference = []

        expected_squashed = "foo"
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=DivergingValuesWarning)
            actual_squashed = squash_column_safely(
                column=column,
                preference=preference,
            )

        assert actual_squashed == expected_squashed

    def test_diverging_with_preference(self):
        column = Series(["foo", "bar", "foo"])
        preference = ["baz", "bar"]

        expected_squashed = "bar"
        with warns(DivergingValuesWarning):
            actual_squashed = squash_column_safely(
                column=column,
                preference=preference,
            )

        assert actual_squashed == expected_squashed

    def test_diverging_without_preference(self):
        column = Series(["foo", "bar", "foo"])
        preference = ["baz", "bazinga"]

        expected_squashed = "foo"
        with warns(DivergingValuesWarning):
            actual_squashed = squash_column_safely(
                column=column,
                preference=preference,
            )

        assert actual_squashed == expected_squashed

class TestWeightedMean:
    def test_simple(self):
        data_frame = DataFrame({
            "weight": [1,2,3],
            "value": [10, 100, 1000],
        })
        expected = 3210
        actual = weighted_mean(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert actual == expected

    def test_empty(self):
        data_frame = DataFrame({
            "weight": [],
            "value": [],
        })
        expected = 0.0
        actual = weighted_mean(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert actual == expected

    def test_nans(self):
        data_frame = DataFrame({
            "weight": [1,2,nan],
            "value": [nan, 100, 1000],
        })
        expected = 200.0
        actual = weighted_mean(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert isclose(actual, expected)

class TestWeightedVariance:
    def test_simple(self):
        data_frame = DataFrame({
            "weight": [1,2,3],
            "value": [10, 100, 1000],
        })
        expected = 44236500
        actual = weighted_variance(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert actual == expected

    def test_empty(self):
        data_frame = DataFrame({
            "weight": [],
            "value": [],
        })
        expected = 0.0
        actual = weighted_variance(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert actual == expected

    def test_nans(self):
        data_frame = DataFrame({
            "weight": [1,2,nan],
            "value": [nan, 100, 1000],
        })
        expected = 20000
        actual = weighted_variance(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert isclose(actual, expected)

class TestWeightedSkewness:
    def test_simple(self):
        data_frame = DataFrame({
            "weight": [1,2,3],
            "value": [10, 100, 1000],
        })
        expected = -0.42590697123040466
        actual = weighted_skewness(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert actual == expected

    def test_empty(self):
        data_frame = DataFrame({
            "weight": [],
            "value": [],
        })
        with warns(RuntimeWarning):
            actual = weighted_skewness(
                data_frame=data_frame,
                value="value",
                weight="weight",
            )
        assert isna(actual)

    def test_nans(self):
        data_frame = DataFrame({
            "weight": [1,2,nan],
            "value": [nan, 100, 1000],
        })
        expected = -0.7071067811865476
        actual = weighted_skewness(
            data_frame=data_frame,
            value="value",
            weight="weight",
        )
        assert isclose(actual, expected)


