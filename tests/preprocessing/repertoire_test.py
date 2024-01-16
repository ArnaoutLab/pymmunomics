from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from pytest import fixture

from pymmunomics.preprocessing.repertoire import (
    count_features,
    get_repertoire_sizes,
    repertoire_from_sequence_table,
)

class TestCountFeatures:
    def test_no_pools_no_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f2", "x",   20 /   20],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_partial_pools_no_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f2", "x",   20 /   20],
                ["pooled", "a", "f1", "1", 1021 / 1031],
                ["pooled", "a", "f1", "3",   10 / 1031],
                ["pooled", "a", "f2", "x",   21 / 1031],
                ["pooled", "a", "f2", "y", 1010 / 1031],
                ["pooled", "b", "f1", "2",  102 /  302],
                ["pooled", "b", "f1", "3",  200 /  302],
                ["pooled", "b", "f2", "x",  202 /  302],
                ["pooled", "b", "f2", "y",  100 /  302],
                ["pooled", "pooled", "f1", "1", 1021 / 1333],
                ["pooled", "pooled", "f1", "2",  102 / 1333],
                ["pooled", "pooled", "f1", "3",  210 / 1333],
                ["pooled", "pooled", "f2", "x",  223 / 1333],
                ["pooled", "pooled", "f2", "y", 1110 / 1333],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"], ["g1", "g2"]],
        )
        assert_frame_equal(actual, expected)

    def test_no_pools_shared_feature_group(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "2",    0 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "1",    0 /  302],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f2", "x",   20 /   20],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            shared_clonotype_feature_groups=["g2"],
        )
        assert_frame_equal(actual, expected)

    def test_partial_pools_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f1", "3",    0 /   20],
                ["b", "a", "f2", "x",   20 /   20],
                ["b", "a", "f2", "y",    0 /   20],
                ["pooled", "a", "f1", "1", 1021 / 1031],
                ["pooled", "a", "f1", "3",   10 / 1031],
                ["pooled", "a", "f2", "x",   21 / 1031],
                ["pooled", "a", "f2", "y", 1010 / 1031],
                ["pooled", "b", "f1", "2",  102 /  302],
                ["pooled", "b", "f1", "3",  200 /  302],
                ["pooled", "b", "f2", "x",  202 /  302],
                ["pooled", "b", "f2", "y",  100 /  302],
                ["pooled", "pooled", "f1", "1", 1021 / 1333],
                ["pooled", "pooled", "f1", "2",  102 / 1333],
                ["pooled", "pooled", "f1", "3",  210 / 1333],
                ["pooled", "pooled", "f2", "x",  223 / 1333],
                ["pooled", "pooled", "f2", "y", 1110 / 1333],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"], ["g1", "g2"]],
            shared_clonotype_feature_groups=["g1"],
        )
        assert_frame_equal(actual, expected)

    def test_absolute_counts(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "count",
            ],
            data=[
                ["a", "a", "f1", "1", 1001],
                ["a", "a", "f1", "3",   10],
                ["a", "a", "f2", "x",    1],
                ["a", "a", "f2", "y", 1010],
                ["a", "b", "f1", "2",  102],
                ["a", "b", "f1", "3",  200],
                ["a", "b", "f2", "x",  202],
                ["a", "b", "f2", "y",  100],
                ["b", "a", "f1", "1",   20],
                ["b", "a", "f2", "x",   20],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            stat="count",
        )
        assert_frame_equal(actual, expected)

    def test_absolute_counts_partial_pools_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "count",
            ],
            data=[
                ["a", "a", "f1", "1", 1001],
                ["a", "a", "f1", "3",   10],
                ["a", "a", "f2", "x",    1],
                ["a", "a", "f2", "y", 1010],
                ["a", "b", "f1", "2",  102],
                ["a", "b", "f1", "3",  200],
                ["a", "b", "f2", "x",  202],
                ["a", "b", "f2", "y",  100],
                ["b", "a", "f1", "1",   20],
                ["b", "a", "f1", "3",    0],
                ["b", "a", "f2", "x",   20],
                ["b", "a", "f2", "y",    0],
                ["pooled", "a", "f1", "1", 1021],
                ["pooled", "a", "f1", "3",   10],
                ["pooled", "a", "f2", "x",   21],
                ["pooled", "a", "f2", "y", 1010],
                ["pooled", "b", "f1", "2",  102],
                ["pooled", "b", "f1", "3",  200],
                ["pooled", "b", "f2", "x",  202],
                ["pooled", "b", "f2", "y",  100],
                ["pooled", "pooled", "f1", "1", 1021],
                ["pooled", "pooled", "f1", "2",  102],
                ["pooled", "pooled", "f1", "3",  210],
                ["pooled", "pooled", "f2", "x",  223],
                ["pooled", "pooled", "f2", "y", 1110],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"], ["g1", "g2"]],
            stat="count",
            shared_clonotype_feature_groups=["g1"],
        )
        assert_frame_equal(actual, expected)

    def test_partial_pools_shared_feature_groups_docs_example(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f1", "3",    0 /   20],
                ["b", "a", "f2", "x",   20 /   20],
                ["b", "a", "f2", "y",    0 /   20],
                ["pooled", "a", "f1", "1", 1021 / 1031],
                ["pooled", "a", "f1", "3",   10 / 1031],
                ["pooled", "a", "f2", "x",   21 / 1031],
                ["pooled", "a", "f2", "y", 1010 / 1031],
                ["pooled", "b", "f1", "2",  102 /  302],
                ["pooled", "b", "f1", "3",  200 /  302],
                ["pooled", "b", "f2", "x",  202 /  302],
                ["pooled", "b", "f2", "y",  100 /  302],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"]],
            shared_clonotype_feature_groups=["g1"],
        )
        assert_frame_equal(actual, expected)

    def test_onehot_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "value",
            ],
            data=[
                ["a", "a", "f1", "1", 1],
                ["a", "a", "f1", "3", 1],
                ["a", "a", "f2", "x", 1],
                ["a", "a", "f2", "y", 1],
                ["a", "b", "f1", "2", 1],
                ["a", "b", "f1", "3", 1],
                ["a", "b", "f2", "x", 1],
                ["a", "b", "f2", "y", 1],
                ["b", "a", "f1", "1", 1],
                ["b", "a", "f1", "3", 0],
                ["b", "a", "f2", "x", 1],
                ["b", "a", "f2", "y", 0],
            ],
        )
        actual = count_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            stat="onehot",
            shared_clonotype_feature_groups=["g1"],
        )
        assert_frame_equal(actual, expected)

class TestGetRepertoireSizes:
    def test_default(self):
        repertoire = DataFrame(
            columns=["g1", "g2", "sample", "other1", "other2"],
            data=[
                ["a", "a", "foo", 1, "foo"],
                ["a", "a", "foo", 2, "foo"],
                ["a", "b", "foo", 3, "bar"],
                ["b", "a", "foo", 4, "foo"],
                ["a", "a", "bar", 5, "bar"],
                ["a", "b", "bar", 6, "foo"],
                ["a", "b", "bar", 7, "bar"],
                ["b", "a", "bar", 8, "foo"],
            ],
        )
        expected = DataFrame(
            index=Index(
                name="sample",
                data=[
                    "foo",
                    "bar",
                ],
            ),
            columns=MultiIndex.from_tuples(
                names=["g1", "g2"],
                tuples=[("a", "a"), ("a", "b"), ("b", "a")],
            ),
            data=[
                [2, 1, 1],
                [1, 2, 1],
            ],
        )
        actual = get_repertoire_sizes(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
        )
        assert_frame_equal(actual, expected, check_like=True)

    def test_id_var(self):
        repertoire = DataFrame(
            columns=["g1", "g2", "id", "sample", "other1", "other2"],
            data=[
                ["a", "a", "foo", "foo", 1, "foo"],
                ["a", "a", "foo", "bar", 2, "foo"],
                ["a", "b", "foo", "bar", 3, "bar"],
                ["b", "a", "foo", "bar", 4, "foo"],
                ["a", "a", "bar", "bar", 5, "bar"],
                ["a", "b", "bar", "bar", 6, "foo"],
                ["a", "b", "bar", "bar", 7, "bar"],
                ["b", "a", "bar", "bar", 8, "foo"],
            ],
        )
        expected = DataFrame(
            index=Index(
                name="id",
                data=[
                    "foo",
                    "bar",
                ],
            ),
            columns=MultiIndex.from_tuples(
                names=["g1", "g2"],
                tuples=[("a", "a"), ("a", "b"), ("b", "a")],
            ),
            data=[
                [2, 1, 1],
                [1, 2, 1],
            ],
        )
        actual = get_repertoire_sizes(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            id_var="id",
        )
        assert_frame_equal(actual, expected, check_like=True)

    def test_clonesize(self):
        repertoire = DataFrame(
            columns=["g1", "g2", "sample", "clonesize", "other1", "other2"],
            data=[
                ["a", "a", "foo", 1, 1, "foo"],
                ["a", "a", "foo", 2, 2, "foo"],
                ["a", "b", "foo", 3, 3, "bar"],
                ["b", "a", "foo", 4, 4, "foo"],
                ["a", "a", "bar", 5, 5, "bar"],
                ["a", "b", "bar", 6, 6, "foo"],
                ["a", "b", "bar", 7, 7, "bar"],
                ["b", "a", "bar", 8, 8, "foo"],
            ],
        )
        expected = DataFrame(
            index=Index(
                name="sample",
                data=[
                    "foo",
                    "bar",
                ],
            ),
            columns=MultiIndex.from_tuples(
                names=["g1", "g2"],
                tuples=[("a", "a"), ("a", "b"), ("b", "a")],
            ),
            data=[
                [3, 3, 4],
                [5, 13, 8],
            ],
        )
        actual = get_repertoire_sizes(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonesize="clonesize"
        )
        assert_frame_equal(actual, expected, check_like=True)

    def test_partial_repertoire_pools(self):
        repertoire = DataFrame(
            columns=["g1", "g2", "sample", "other1", "other2"],
            data=[
                ["a", "a", "foo", 1, "foo"],
                ["a", "a", "foo", 2, "foo"],
                ["a", "b", "foo", 3, "bar"],
                ["b", "a", "foo", 4, "foo"],
                ["a", "a", "bar", 5, "bar"],
                ["a", "b", "bar", 6, "foo"],
                ["a", "b", "bar", 7, "bar"],
                ["b", "a", "bar", 8, "foo"],
            ],
        )
        expected = DataFrame(
            index=Index(
                name="sample",
                data=[
                    "foo",
                    "bar",
                ],
            ),
            columns=MultiIndex.from_tuples(
                names=["g1", "g2"],
                tuples=[
                    ("a", "a"), ("a", "b"), ("b", "a"), ("pooled", "a"), ("pooled", "b")
                ],
            ),
            data=[
                [2, 1, 1, 3, 1],
                [1, 2, 1, 2, 2],
            ],
        )
        actual = get_repertoire_sizes(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            partial_repertoire_pools=[[], ["g1"]],
        )
        assert_frame_equal(actual, expected, check_like=True)

    def test_partial_pooles_clone_sizes_docs_example(self):
        repertoire = DataFrame(
            columns=["g1", "g2", "sample", "clonesize"],
            data=[
                ["a", "a", "foo", 1],
                ["a", "a", "foo", 2],
                ["a", "b", "foo", 3],
                ["b", "a", "foo", 4],
                ["a", "a", "bar", 5],
                ["a", "b", "bar", 6],
                ["a", "b", "bar", 7],
                ["b", "a", "bar", 8],
            ],
        )
        expected = DataFrame(
            index=Index(
                name="sample",
                data=[
                    "foo",
                    "bar",
                ],
            ),
            columns=MultiIndex.from_tuples(
                names=["g1", "g2"],
                tuples=[
                    ("a", "a"), ("a", "b"), ("b", "a"), ("pooled", "a"), ("pooled", "b")
                ],
            ),
            data=[
                [3, 3, 4, 7, 3],
                [5, 13, 8, 13, 13],
            ],
        )
        actual = get_repertoire_sizes(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"]],
        )
        assert_frame_equal(actual, expected, check_like=True)

class TestRepertoireFromSequenceTable:
    @fixture
    def sequence_table(self):
        return DataFrame(
            columns=["c1", "c2", "c3", "other1", "other2"],
            data=[
                ["foo", "foo", "foo", "other1_1", "other2_1"],
                ["foo", "foo", "bar", "other1_1", "other2_1"],
                ["foo", "bar", "foo", "other1_1", "other2_1"],
                ["foo", "bar", "bar", "other1_1", "other2_1"],
                ["bar", "foo", "foo", "other1_1", "other2_1"],
                ["bar", "foo", "bar", "other1_1", "other2_1"],
                ["bar", "bar", "foo", "other1_1", "other2_1"],
                ["bar", "bar", "bar", "other1_1", "other2_1"],
                ["foo", "foo", "bar", "other1_2", "other2_1"],
                ["bar", "foo", "foo", "other1_1", "other2_2"],
                ["foo", "foo", "foo", "other1_2", "other2_2"],
                ["foo", "foo", "foo", "other1_2", "other2_3"],
                ["foo", "bar", "bar", "other1_1", "other2_1"],
                ["foo", "bar", "bar", "other1_1", "other2_1"],
            ]
        )

    def test_single_clonotype_column(self, sequence_table):
        clonotype_columns = ["c1"]

        sort_cols = ["c1"]
        sorted_cols = ["c1", "clone_size", "clone_fraction"]
        expected_repertoire = (
            DataFrame(
                columns=["c1", "clone_size", "clone_fraction"],
                data=[
                    ["foo", 9, 9/14],
                    ["bar", 5, 5/14],
                ],
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        actual_repertoire = (
            repertoire_from_sequence_table(
                sequence_table=sequence_table,
                clonotype_columns=clonotype_columns,
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        assert_frame_equal(actual_repertoire, expected_repertoire)

    def test_multiple_clonotype_column(self, sequence_table):
        clonotype_columns = ["c1", "c2"]

        sort_cols = ["c1", "c2"]
        sorted_cols = ["c1", "c2", "clone_size", "clone_fraction"]
        expected_repertoire = (
            DataFrame(
                columns=["c1", "c2", "clone_size", "clone_fraction"],
                data=[
                    ["foo", "foo", 5, 5/14],
                    ["foo", "bar", 4, 4/14],
                    ["bar", "foo", 3, 3/14],
                    ["bar", "bar", 2, 2/14],
                ],
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        actual_repertoire = (
            repertoire_from_sequence_table(
                sequence_table=sequence_table,
                clonotype_columns=clonotype_columns,
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        assert_frame_equal(actual_repertoire, expected_repertoire)

    def test_count_name_frequency_name(self, sequence_table):
        clonotype_columns = ["c1", "c2", "c3"]
        count_name = "count"
        frequency_name = "frequency"

        sort_cols = ["c1", "c2", "c3"]
        sorted_cols = ["c1", "c2", "c3", "count", "frequency"]
        expected_repertoire = (
            DataFrame(
                columns=["c1", "c2", "c3", "count", "frequency"],
                data=[
                    ["foo", "foo", "foo", 3, 3/14],
                    ["foo", "foo", "bar", 2, 2/14],
                    ["foo", "bar", "foo", 1, 1/14],
                    ["foo", "bar", "bar", 3, 3/14],
                    ["bar", "foo", "foo", 2, 2/14],
                    ["bar", "foo", "bar", 1, 1/14],
                    ["bar", "bar", "foo", 1, 1/14],
                    ["bar", "bar", "bar", 1, 1/14],
                ],
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        actual_repertoire = (
            repertoire_from_sequence_table(
                sequence_table=sequence_table,
                clonotype_columns=clonotype_columns,
                count_name=count_name,
                frequency_name=frequency_name,
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        assert_frame_equal(actual_repertoire, expected_repertoire)

    def test_other_agg(self, sequence_table):
        clonotype_columns = ["c1", "c2"]
        other_agg = {"other1": set, "other2": tuple}

        sort_cols = ["c1", "c2"]
        sorted_cols = ["c1", "c2", "clone_size", "clone_fraction", "other1", "other2"]
        expected_repertoire = (
            DataFrame(
                columns=["c1", "c2", "clone_size", "clone_fraction", "other1", "other2"],
                data=[
                    ["foo", "foo", 5, 5/14, {"other1_1", "other1_2"}, ("other2_1", "other2_1", "other2_1", "other2_2", "other2_3")],
                    ["foo", "bar", 4, 4/14, {"other1_1",}, ("other2_1", "other2_1", "other2_1", "other2_1")],
                    ["bar", "foo", 3, 3/14, {"other1_1",}, ("other2_1", "other2_1", "other2_2")],
                    ["bar", "bar", 2, 2/14, {"other1_1",}, ("other2_1", "other2_1")],
                ],
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        actual_repertoire = (
            repertoire_from_sequence_table(
                sequence_table=sequence_table,
                clonotype_columns=clonotype_columns,
                other_agg=other_agg,
            )
            .sort_values(sort_cols)
            .reset_index(drop=True)
            [sorted_cols]
        )

        assert_frame_equal(actual_repertoire, expected_repertoire)


