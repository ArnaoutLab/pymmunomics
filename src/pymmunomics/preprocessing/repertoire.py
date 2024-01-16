from typing import Iterable, Literal, Mapping, Sequence, Union

from pandas import concat, DataFrame

from pymmunomics.helper.pandas_helpers import (
    concat_partial_groupby_apply,
    concat_pivot_pipe_melt,
    concat_weighted_value_counts,
)

def count_features(
    repertoire: DataFrame,
    repertoire_groups: Sequence[str],
    clonotype_features: Sequence[str],
    clonesize: Union[str, None] = None,
    partial_repertoire_pools: Union[Iterable[Sequence[str]], None] = None,
    stat: Literal["count", "frequency", "onehot"] = "frequency",
    shared_clonotype_feature_groups: Union[Sequence[str], None] = None,

):
    """Counts various clonotype feature occurrences in repertoires.

    Parameters
    ----------
    repertoire:
        Table of clonotypes and their clone sizes as well as columns
        indicating repertoire membership.
    repertoire_groups:
        Columns indicating repertoire memberships.
    clonotype_features:
        Columns representing different clonotype features or
        combinations thereof.
    clonesize:
        Column listing the clone sizes in corresponding to the
        clonotypes and repertoires.
    partial_repertoire_pools:
        If specified, counts are calculated and concatenated once for
        each member of this list with the indicated columns' values
        replaced by the value "pooled" effectively combining repertoires
        with same values in these columns.
    stat:
        Determines how features are counted.

        - 'count': provides absolute counts
        - 'frequency': provides counts normalized by repertoire sizes
        - 'onehot': provides one-hot encoding indicating
          presence/absences of values; zeros value can only be achieved
          when a valid argument for shared_clonotype_feature_groups is
          provided.
    shared_clonotype_feature_groups:
        If provided, all repertoires with the same value in all but these
        group columns are forced to have frequencies for the same clonotype
        feature values per feature group. Missing counts are set to 0.

    Returns
    -------
    feature_counts:
        Table listing frequency per repertoire, feature and feature
        value. Columns are: `repertoire_groups` which indicate
        repertoire membership of a count, "feature" which contains one
        of `clonotype_features` indicating which clonotype feature the
        row counts, "feature_value" indicating a count's corresponding
        counted value of the feature and finally, "frequency", or
        "count", depending on the value of `relative_counts`, which
        contain the actual counts.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.preprocessing.repertoire import count_features
    >>> 
    >>> repertoire = pd.DataFrame(
    ...     columns=[
    ...         "g1", "g2", "f1", "f2", "clonesize",
    ...     ],
    ...     data=[
    ...         ["a", "a", 1, "x", 1],
    ...         ["a", "b", 2, "x", 2],
    ...         ["a", "a", 3, "y", 10],
    ...         ["b", "a", 1, "x", 20],
    ...         ["a", "b", 2, "y", 100],
    ...         ["a", "b", 3, "x", 200],
    ...         ["a", "a", 1, "y", 1000],
    ...     ],
    ... )
    >>> count_features(
    ...     repertoire=repertoire,
    ...     repertoire_groups=["g1", "g2"],
    ...     clonotype_features=["f1", "f2"],
    ...     clonesize="clonesize",
    ...     partial_repertoire_pools=[[], ["g1"]],
    ...     shared_clonotype_feature_groups=["g1"],
    ... )
            g1 g2 feature feature_value  frequency
    0        a  a      f1             1   0.990109
    1        a  a      f1             3   0.009891
    2        a  a      f2             x   0.000989
    3        a  a      f2             y   0.999011
    4        a  b      f1             2   0.337748
    5        a  b      f1             3   0.662252
    6        a  b      f2             x   0.668874
    7        a  b      f2             y   0.331126
    8        b  a      f1             1   1.000000
    9        b  a      f1             3   0.000000
    10       b  a      f2             x   1.000000
    11       b  a      f2             y   0.000000
    12  pooled  a      f1             1   0.990301
    13  pooled  a      f1             3   0.009699
    14  pooled  a      f2             x   0.020369
    15  pooled  a      f2             y   0.979631
    16  pooled  b      f1             2   0.337748
    17  pooled  b      f1             3   0.662252
    18  pooled  b      f2             x   0.668874
    19  pooled  b      f2             y   0.331126
    """
    if partial_repertoire_pools is None:
        partial_repertoire_pools = [[]]
    count_col, normalize = {
        "count": ("count", False),
        "frequency": ("frequency", True),
        "onehot": ("value", False),
    }[stat]
    
    counts = (
        concat_partial_groupby_apply(
            data_frame=repertoire,
            func=concat_weighted_value_counts,
            by=repertoire_groups,
            pooled=partial_repertoire_pools,
            subsets=clonotype_features,
            weight=clonesize,
            normalize=normalize,
        )
        .reset_index()
        .astype({"values": str})
        .rename(columns={
            "values": "feature_value",
            "columns": "feature",
            {True: "frequency", False: "count"}[normalize]: count_col,
        })
    )
    
    if shared_clonotype_feature_groups is not None:
        groupby_cols = [
            "feature",
            *[
                col for col in repertoire_groups
                if col not in shared_clonotype_feature_groups
            ]
        ]
        repertoire_feature_table = (
            counts
            .groupby(groupby_cols)
            .apply(
                concat_pivot_pipe_melt,
                DataFrame.fillna,
                values=[count_col],
                columns="feature_value",
                index=shared_clonotype_feature_groups,
                value=0,
            )
            .reset_index()
            [[
                *repertoire_groups,
                "feature",
                "feature_value",
                count_col,
            ]]
        )
    else:
        repertoire_feature_table = counts
    if stat == "onehot":
        repertoire_feature_table[count_col] = (
            repertoire_feature_table[count_col]
            .map(lambda val: int(val > 0))
        )

    if stat != "frequency":
        repertoire_feature_table[count_col] = (
            repertoire_feature_table[count_col]
            .astype(int)
        )

    return (
        repertoire_feature_table
        .sort_values([*repertoire_groups, "feature", "feature_value"])
        .reset_index(drop=True)
    )

def get_repertoire_sizes(
    repertoire: DataFrame,
    repertoire_groups: Sequence[str],
    id_var: str = "sample",
    clonesize: Union[str, None] = None,
    partial_repertoire_pools: Union[Iterable[Sequence[str]], None] = None,
):
    """Obtains sequence repertoire sizes in various groupings.

    Parameters
    ----------
    repertoire:
        Table of clonotypes and their clone sizes as well as columns
        indicating repertoire membership.
    repertoire_groups:
        Columns by which to group repertoires.
    id_var:
        Indicates which column identifies individual repertoires within
        repertoire groups.
    clonesize:
        If provided, repertoire is sum of clonesizes in rows with given
        value in `id_var` within groups determined by value
        combinations in `repertoire_groups`. If None, each row has an
        effective clone size of 1.
    partial_repertoire_pools:
        If specified, repertoire sizes are calculated and concatenated
        once for each member of this list with the indicated columns'
        values replaced by the value "pooled" effectively combining
        repertoires with same values in these columns.

    Returns
    -------
    repertoire_sizes:
        Table listing repertoire sizes. Index consists of values in
        `id_var` column, and columns are indexed by combinations of
        values in `repertoire_groups`, replacing some values with
        "pooled" if `partial_repertoire_pools` are specified. Each value
        is the indexed repertoire size within the column's group.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.preprocessing.repertoire import get_repertoire_sizes
    >>> 
    >>> repertoire = pd.DataFrame(
    ...     columns=["g1", "g2", "sample", "clonesize"],
    ...     data=[
    ...         ["a", "a", "foo", 1],
    ...         ["a", "a", "foo", 2],
    ...         ["a", "b", "foo", 3],
    ...         ["b", "a", "foo", 4],
    ...         ["a", "a", "bar", 5],
    ...         ["a", "b", "bar", 6],
    ...         ["a", "b", "bar", 7],
    ...         ["b", "a", "bar", 8],
    ...     ],
    ... )
    >>> get_repertoire_sizes(
    ...     repertoire=repertoire,
    ...     repertoire_groups=["g1", "g2"],
    ...     clonesize="clonesize",
    ...     partial_repertoire_pools=[[], ["g1"]],
    ... )
    g1      a      b pooled    
    g2      a   b  a      a   b
    sample                     
    bar     5  13  8     13  13
    foo     3   3  4      7   3
    """
    sizes = (
        count_features(
            repertoire=repertoire,
            repertoire_groups=repertoire_groups,
            clonotype_features=[id_var],
            clonesize=clonesize,
            partial_repertoire_pools=partial_repertoire_pools,
            stat="count",
        )
        .rename(columns={"feature_value": id_var})
        .pivot(index=id_var, columns=repertoire_groups, values="count")
        .fillna(0)
        .astype(int)
    )
    return sizes

def repertoire_from_sequence_table(
    sequence_table: DataFrame,
    clonotype_columns: Sequence[str],
    count_name: str = "clone_size",
    frequency_name: str = "clone_fraction",
    other_agg: Union[dict, None] = None,
) -> DataFrame:
    """Converts observations into values with counts and frequencies.

    Parameters
    ----------
    sequence_table:
        Table of sequence observations. Lists the same clonotype
        repeatedly each time it is observed.
    clonotype_columns:
        Columns whose value combinations identify and distinguish
        clonotypes
    count_name:
        Name of column for counts.
    frequency_name:
        Name of colunn for normalized counts.
    other_agg:
        Maps other columns that are not in clonotype_columns to
        aggregation functions to aggregate over same clonotype groups.

    Returns
    -------
    repertoire:
        A table listing clonotypes and their counts, normalized counts
        and possibly aggregations of other columns across the
        clonotypes.
    """
    counts = (
        sequence_table[clonotype_columns]
        .value_counts()
        .to_frame()
        .rename(columns={0: count_name})
    )
    frequencies = (
        sequence_table[clonotype_columns]
        .value_counts(normalize=True)
        .to_frame()
        .rename(columns={0: frequency_name})
    )
    join_frames = [counts, frequencies]
    if other_agg is not None:
        join_frames.append(
            sequence_table.groupby(clonotype_columns).agg(other_agg)
        )
    repertoire = concat(join_frames, axis=1).reset_index()
    return repertoire
