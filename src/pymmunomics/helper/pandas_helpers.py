from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from warnings import warn

from numpy import allclose, empty
from numpy.typing import ArrayLike
from pandas import concat, DataFrame, read_csv, Series
from pandas.testing import assert_frame_equal

from pymmunomics.helper.exception import (
    AmbiguousValuesWarning,
    DivergingValuesWarning,
    InvalidArgumentError,
)
from pymmunomics.helper.generic_helpers import chain_update

def agg_first_safely(
    series: Series,
    dropna: bool = True,
):
    if series.nunique(dropna=dropna) > 1:
        warn(AmbiguousValuesWarning(
            "Aggregating values %s using first row's value: %s; (series name: %s)"
            % (set(series), series.iloc[0:1], series.name)
        ))
    return series.iloc[0]

def apply_zipped(
    data_frame: DataFrame,
    keys: Sequence[str],
    func: Callable,
    *args,
    unpack: bool = True,
    **kwargs,
) -> list:
    """Efficiently applies function to tuples of values from columns.

    Parameters
    ----------
    data_frame:
        Data to which to apply function.
    keys:
        Columns whose values to zip and pass to function.
    func:
        Called on each combination of values in the specified columns.
    unpack:
        Determines whether or not value combinations should be passed as
        tuples or as individual positional arguments.
    args, kwargs:
        Passed to func at every call.

    Returns
    -------
    return_values:
        A list of return values corresponding to the function calls.
    """
    zipped_columns = zip(*[data_frame[key] for key in keys])
    if unpack:
        return [
            func(*vals, *args, **kwargs)
            for vals in zipped_columns
        ]
    else:
        return [
            func(vals, *args, **kwargs)
            for vals in zipped_columns
        ]

def assert_groups_equal(
    data_frame: DataFrame,
    groupby_kwargs: Dict,
    group_pipe: Union[Callable, None] = None,
    assert_frame_equal_kwargs: Union[Dict, None] = None,
):
    """Checks equality of all-against-one groups in data_frame.

    Parameters
    ----------
    data_frame:
        Data to group and test equality of all-against-one group.
    groupby_kwargs:
        Keyword arguments for `pandas.DataFrame.groupby`.
    group_pipe:
        Function applied to each group before testing equality.
    assert_frame_equal_kwargs:
        Keyword arguments for `pandas.testing.assert_frame_equal` used
        for each pair of groups.

    Raises
    ------
    AssertionError:
        Propagated from `pandas.testing.assert_frame_equal` when a
        comparison fails.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.helper.pandas_helpers import assert_groups_equal
    >>> 
    >>> data_frame = pd.DataFrame(
    ...     index=[1, 2, 2, 1, 1, 2],
    ...     columns=["g1", "g2", "val1", "val2"],
    ...     data=[
    ...         ["a", "a", 1, 1],
    ...         ["a", "a", 2, 2],
    ...         ["a", "b", 1, -1],
    ...         ["a", "b", 2, 2],
    ...         ["b", "a", 1, 1],
    ...         ["b", "a", 2, 2],
    ...     ],
    ... )
    >>> assert_groups_equal(
    ...     data_frame=data_frame,
    ...     groupby_kwargs={"by": ["g1", "g2"]},
    ...     group_pipe=lambda df: df[["val1"]].reset_index(drop=True),
    ... )
    >>> assert_groups_equal(
    ...     data_frame=data_frame,
    ...     groupby_kwargs={"by": ["g1", "g2"]},
    ...     group_pipe=lambda df: df[["val2"]].reset_index(drop=True),
    ... )
    Traceback (most recent call last):
        ...
    AssertionError: DataFrame.iloc[:, 0] (column name="val2") are different
    """
    if group_pipe is None:
        group_pipe = lambda df: df
    if assert_frame_equal_kwargs is None:
        assert_frame_equal_kwargs = {}

    groupby = data_frame.groupby(**groupby_kwargs)
    groups = list(groupby.groups)
    first = group_pipe(groupby.get_group(groups[0]))
    for group in groups[1:]:
        other = group_pipe(groupby.get_group(group))
        assert_frame_equal(first, other, **assert_frame_equal_kwargs)

def column_combinations(data_frame: DataFrame, columns: Sequence[str]):
    """Obtains set of value combinations in columns.

    Parameters
    ----------
    data_frame:
        Data from which to extract combinations.
    columns:
        Columns whose value-combinations to extract.

    Returns
    -------
    combinations:
        A set of tuples of combinations of values in specified columns.
    """
    return set(zip(*[data_frame[col] for col in columns]))

def concat_partial_groupby_apply(
    data_frame: DataFrame,
    func: Union[Callable, list[Callable]],
    by: Sequence[str],
    *func_args,
    pooled: Iterable[Sequence[str]] = [[]],
    pooled_val: str = "pooled",
    func_keys: Union[Sequence, None] = None,
    col_names: Union[list, None] = None,
    **func_kwargs,
) -> DataFrame:
    """Applies func to partial groupings data.

    Parameters
    ----------
    data_frame:
        Data to group and apply function to.
    func:
        Must take a group of `data_frame` grouped by `by` and return a
        ``DataFrame``. Additional arguments and keyword argument can be
        passed using `*func_args` and `*func_kwargs`. If multiple
        callables, the functions are applied independently and their
        results are concatenated along axis=1 per partially pooled
        group.
    by:
        The columns to group by if no columns were pooled.
    pooled:
        The various subsets of `by` whose subgroups in the grouped
        `data_frame` should be pooled for the different applications of
        `func` on the resulting partially pooled groups.
    pooled_val:
        The value to assign to the columns defined by members of
        `pooled` whenever their subgroups are pooled.
    func_keys:
        If provided, an additional index level is prepended to the
        columns with these values corresponding to the results of each
        of the applied functions.
    col_names:
        Names for the levels in the resulting hierarchical column index.
    *func_args, **func_kwargs:
        Additional positional and keyword arguments to forward to func
        in addition to each data group.

    Returns
    -------
    func_results:
        Concatenation of results of func applied to the various data
        groupings. Index lists the values for columns listed in `by`
        indicating the group func was applied to. Pooled columns will
        have `pooled_val` in this index.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.helper.pandas_helpers import concat_partial_groupby_apply
    >>> 
    >>> data_frame = pd.DataFrame(
    ...     columns=["group1", "group2", "val"],
    ...     data=[
    ...         ["a", "a", 1],
    ...         ["a", "a", 2],
    ...         ["b", "a", 3],
    ...         ["b", "a", 5],
    ...         ["a", "b", 7],
    ...         ["a", "b", 11],
    ...         ["b", "a", 13],
    ...     ],
    ... )
    >>> concat_partial_groupby_apply(
    ...     data_frame=data_frame,
    ...     func=lambda df: df[["val"]].prod(),
    ...     by=["group1", "group2"],
    ...     pooled=[[], ["group2"], ["group1", "group2"]],
    ... )
                     val
    group1 group2       
    a      a           2
           b          77
    b      a         195
    a      pooled    154
    b      pooled    195
    pooled pooled  30030
    """
    if type(func) != list:
        func = [func]
    concat_vertical = []
    for pcols in pooled:
        concat_horizontal = []
        for func_ in func:
            pooled_data_frame = (
                data_frame
                .drop(columns=pcols)
                .assign(**{col: pooled_val for col in pcols})
            )
            horizontal_item = (
                pooled_data_frame
                .groupby(by)
                .apply(func_, *func_args, **func_kwargs)
            )
            concat_horizontal.append(horizontal_item)
        vertical_item = concat(
            concat_horizontal,
            axis=1,
            keys=func_keys,
            names=col_names,
        )
        concat_vertical.append(vertical_item)
    result = concat(concat_vertical)
    return result


def concat_pivot_pipe_melt(
    data_frame: DataFrame,
    func: Callable,
    values: Sequence[str],
    columns: Union[str, Sequence[str]],
    *func_args,
    index: Optional[Union[str, Sequence[str]]] = None,
    **func_kwargs,
) -> DataFrame:
    """Pipes pivotted data into func and melts result.

    Parameters
    ----------
    data_frame:
        Data to pivot, pipe and melt.
    func:
        Must take pivotted `data_frame` with columns partially selected
        for one of `values` at a time as input and return a
        ``DataFrame``.
    values, columns, index:
        Used as argument for same-named parameters in
        ``pandas.DataFrame.pivot``. If single value column, then a
        sequence of length 1 (ex.: ["value_col"]) must be used.
    *func_args, **func_kwargs:
        Additional positional and keyword arguments to forward to func
        in addition to the pivotted data.

    Returns
    -------
    func_results:
        Concatenation (along axis=1) of melted return values of `func`
        applied to pivotted data with partially selected columns for
        members of `values` one at a time. Index consists of
        combinations of values of `columns` and either values of
        original index of `data_frame`, or values in `index` columns if
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pymmunomics.helper.pandas_helpers import concat_pivot_pipe_melt
    >>> 
    >>> data_frame = pd.DataFrame(
    ...     columns=["idx", "col", "val1", "val2"],
    ...     data=[
    ...         ["a", "a", 1, 1],
    ...         ["a", "b", 2, np.nan],
    ...         ["a", "c", 3, 3],
    ...         ["b", "a", 4, 4],
    ...         ["b", "b", 5, np.nan],
    ...     ],
    ... )
    >>> data_frame
      idx col  val1  val2
    0   a   a     1   1.0
    1   a   b     2   NaN
    2   a   c     3   3.0
    3   b   a     4   4.0
    4   b   b     5   NaN
    >>> concat_pivot_pipe_melt(
    ...     data_frame=data_frame,
    ...     func=pd.DataFrame.fillna,
    ...     values=["val1", "val2"],
    ...     columns="col",
    ...     index="idx",
    ...     value=0,
    ... )
             val1  val2
    idx col            
    a   a     1.0   1.0
        b     2.0   0.0
        c     3.0   3.0
    b   a     4.0   4.0
        b     5.0   0.0
        c     0.0   0.0
    """
    pivotted = data_frame.pivot(index=index, columns=columns, values=values)
    piped = func(pivotted, *func_args, **func_kwargs)
    melted = concat(
        [
            piped[val]
            .melt(value_name=val, ignore_index=(index is None))
            .set_index(columns, append=(index is not None))
            for val in values
        ],
        axis=1,
    )
    return melted.sort_index()


def concat_weighted_value_counts(
    data_frame: DataFrame,
    subsets: Iterable[Union[str, List[str]]],
    weight: Union[str, None] = None,
    normalize: bool = False,
) -> DataFrame:
    """Counts weighted number of occurrences of value combinations.

    Parameters
    ----------
    data_frame:
        Table whose values to count.
    subsets:
        Column combinations whose values to count.
    weight:
        Effective count of each row's value combination.
    normalize:
        If true, return relative frequencies instead of absolute counts.
        The counts for each column combination are normalized
        independently.

    Returns
    -------
    feature_counts:
        A table indexed by column combinations (index level 'columns')
        and their corresponding values (index level 'values') with
        single column 'count' (or 'frequency' if `normalize == True`),
        which contains the sum of the effective counts of the indexed
        value combination (normalized by the sum of all effective
        counts).

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.helper.pandas_helpers import concat_weighted_value_counts
    >>> data_frame = pd.DataFrame(
    ...     columns=["column_1", "column_2", "weight"],
    ...     data=[
    ...         ["a", "x", 1],
    ...         ["a", "x", 10],
    ...         ["b", "x", 100],
    ...         ["b", "y", 1000],
    ...         ["c", "y", 10000],
    ...     ],
    ... )
    >>> concat_weighted_value_counts(
    ...     data_frame=data_frame,
    ...     subsets=["column_2", ["column_1", "column_2"]],
    ...     weight="weight",
    ...     normalize=True,
    ... )
                                 frequency
    columns              values           
    column_2             x        0.009990
                         y        0.990010
    (column_1, column_2) (a, x)   0.000990
                         (b, x)   0.009000
                         (b, y)   0.090001
                         (c, y)   0.900009
    """
    counts_list = []
    if weight is None:
        # produces nonexistent column header
        weight = "_".join(data_frame.columns) + "_"
        data_frame = data_frame.assign(**{weight: 1})

    for subset in subsets:
        counts_list.append(
            data_frame
            .groupby(subset)
            [[weight]]
            .sum()
            .pipe(lambda df: df.assign(
                values=df.index.to_flat_index(),
                columns=(
                    [
                        subset
                        if type(subset) == str
                        else tuple(subset)
                    ]
                    * len(df)
                ),
            ))
            .set_index(["columns", "values"])
            .rename(columns={
                weight: "frequency" if normalize else "count"
            })
        )
        if normalize:
            counts_list[-1] /= data_frame[weight].sum()
    counts = concat(counts_list)
    return counts

def pipe_assign_from_func(
    data_frame: DataFrame,
    names: Union[str, Sequence[str]],
    pipe_func: Callable,
    inplace: bool = False,
    **kwargs,
) -> DataFrame:
    """Assigns column(s) from function applied to data.

    Parameters
    ----------
    data_frame:
        Data to which to apply function and append column(s).
    names:
        Name(s) of columns to assign.
    pipe_func:
        Function to apply to data to obtain new columns data.
    inplace:
        Determines whether or not data is modified in place.
    kwargs:
        Passed to `pipe_func`.

    Returns
    -------
    new_data_frame:
        The old `data_frame` plus new columns named `keys` with values
        filled in by func applied to `data_frame`.
    """
    if inplace:
        data_frame_ = data_frame
    else:
        data_frame_ = data_frame.copy()
    data_frame_[names] = pipe_func(data_frame, **kwargs)
    return data_frame_

# def read_as_tuples(
#     filepath: str,
#     columns: Sequence[str],
#     read_func: Callable = read_csv,
#     read_kwargs: Union[dict, None] = None,
# ) -> list:
#     """Reads columns into list of tuples.

#     Parameters
#     ----------
#     filepath:
#         Path to file containing columns of interest.
#     columns:
#         Columns of interest.
#     read_func:
#         Reads filepath into ``pandas.DataFrame``.
#     read_kwargs:
#         Passed to `read_func`.

#     Returns
#     -------
#     tuples:
#         List of tuples of the value combinations in the columns of
#         interest.
#     """
#     if read_kwargs is None:
#         read_kwargs = {}
#     return list(
#         read_func(filepath, **read_kwargs)
#         [columns]
#         .apply(tuple, axis=1)
#     )

def read_mapping(
    filepath: str,
    key: Union[str, Sequence[str]],
    value: str,
    read_func: Callable = read_csv,
    read_kwargs: Union[dict, None] = None,
) -> dict:
    """Reads file into dictionary mapping key to values columns.

    Parameters
    ----------
    filepath:
        Path to file containing mapped keys and values.
    key:
        Column(s) containing the keys.
    value:
        Column containing the values.
    read_func:
        Reads filepath into ``pandas.DataFrame``.
    read_kwargs:
        Passed to `read_func`.

    Note
    ----
    Duplicate mappings are uniqued without guarantees.

    Returns
    -------
    mapping: dict
        Maps key column entries to corresponding value column entries.
    """
    if read_kwargs is None:
        read_kwargs = {}
    if type(key) == str:
        subset = [key, value]
    else:
        subset = [*key, value]
    mapping = (
        read_func(filepath, **read_kwargs)
        [subset]
        .dropna()
        .set_index(key)
        .to_dict()
        [value]
    )
    return mapping

def read_combine_mappings(
    filepaths: Sequence[str],
    keys: Sequence[str],
    values: Sequence[str],
    read_funcs: Sequence[Callable],
    read_kwargs: Union[Sequence[Union[dict, None]], None],
) -> dict:
    """Reads files into dictionary mapping key to values columns.

    Parameters
    ----------
    filepaths:
        Paths to files containing mapped keys and values.
    keys:
        Each file's column containing the keys.
    values:
        Each file's column containing the values.
    read_funcs:
        For each file, reads filepath into ``pandas.DataFrame``.
    read_kwargs:
        For each file, passed to corresponding member in `read_funcs`.

    Note
    ----
    - Duplicate mappings within files are uniqued without guarantees.
    - Duplicate mappings in subsequent files take precedence.
    - Transitive mappings in subsequent files are added individually and
      combined (i.e. if first file maps x -> y and second maps y -> z,
      then in the returned dictionary there will be the mappings y -> z
      and x -> z).
    - Duplicate mappings in subsequent files are resolved before
      transitive mappings.

    Returns
    -------
    mapping: dict
        Maps key column entries to corresponding value column entries.
    """
    zip_lists = [filepaths, keys, values, read_funcs, read_kwargs]
    if len(set(map(len, zip_lists))) > 1:
        raise InvalidArgumentError(
            "Must provide equal numbers of filepaths, key columns,"
            " value columns, read functions, and read function keyword"
            " arguments."
        )
    mappings = []
    for filepath, key, value, read_func, read_kwargs_ in zip(*zip_lists):
        mapping = read_mapping(
            filepath=filepath,
            key=key,
            value=value,
            read_func=read_func,
            read_kwargs=read_kwargs_,
        )
        mappings.append(mapping)
    combined_mappings = chain_update(mappings=mappings)
    return combined_mappings

def remove_duplicated_rows(
    data_frame: DataFrame,
    identity_columns: Sequence[str],
    value_columns: Sequence[str],
    equal_nan: bool=False,
) -> tuple[DataFrame, DataFrame]:
    """Removes rows corresponding to duplicated data.

    Parameters
    ----------
    data_frame:
        Table with columns identity_columns and value_columns.
    identity_columns:
        Names of identity columns. These columns determine what is
        considered duplicate data.
    value_columns:
        Names of value columns. These columns are tested for equality
        among duplicated rows. Must be numeric columns.
    equal_nan:
        See numpy.allclose. Used when comparing rows of valus in
        value_columns To test for uniqueness of values per combination
        of items in identity_columns.

    Returns
    -------
    A tuple consisting of:
    0 - uniqued:
        Same as data_frame, except rows with duplicate identity column
        value combinations removed. First data row for any given
        identity column value combination is kept.
    1 - duplicated:
        The portion of data_frame that was removed.

    Notes
    -----
    Will warn if duplicated rows didn't agree in value_columns.
    """
    is_duplicate = data_frame[identity_columns].duplicated()
    grouped = data_frame.groupby(identity_columns)
    for duplicate, df in grouped:
        if len(df) > 0:
            all_values = df[value_columns].to_numpy()
            first_row = all_values[0]
            diverging_values = False
            for row in all_values[1:]:
                if not allclose(first_row, row, equal_nan=equal_nan):
                    diverging_values = True

            if diverging_values:
                warn(DivergingValuesWarning(
                    "Values for %s diverged unexpectedly; only first"
                    " measurement is used: %s." % (duplicate, df)
                ))
    return (data_frame.loc[~is_duplicate], data_frame.loc[is_duplicate])

def squash_column_safely(column: Series, preference: list=[]):
    """Reduces column into single value.

    Parameters
    ----------
    column:
        Column to reduce.
    preference:
        List of values to choose when ambiguities are present in order
        of preference.

    Returns
    -------
    The first value of column if unambiguous. If column contains
    multiple unique values, then the first item in preference that is
    also in the column is returned. If none of the items preference are
    found, then the first item is returned.

    Notes
    -----
    Warns if column contains more than one unique value.
    """
    uniqued = set(column.unique())
    if len(uniqued) > 1:
        for item in preference:
            if item in uniqued:
                squashed_value = item
                break
        else:
            squashed_value = column.iloc[0]
        warn(DivergingValuesWarning(
            f"More than one distinct value in column:"
            f" {uniqued}; squashed into single value:"
            f" {squashed_value}."
        ))
    else:
        squashed_value = column.iloc[0]
    return squashed_value

def weighted_mean(
    data_frame: DataFrame,
    value: str,
    weight: str,
):
    """Calculates weigthed mean of column values.

    Parameters
    ----------
    data_frame:
        The table containing weights and values.
    value:
        Column containing values.
    weight:
        Column containing weights corresponding to values.

    Returns
    -------
    result
        The resulting weighted mean defined as:
            sum(weights * values)
    """
    data_frame_ = data_frame[
        ~(data_frame[[weight, value]].isna().any(axis=1))
    ]
    return (data_frame_[value] * data_frame_[weight]).sum()

def weighted_variance(data_frame, value, weight):
    """Calculates variance from values and weights.
    
    Parameters
    ----------
    data_frame:
        The table containing weights and values.
    value:
        Column containing values.
    weight:
        Column containing weights corresponding to values.

    Returns
    -------
    result
        The resulting variance defined as:
            sum(weights * ((values - mean) ** 2))
    """
    data_frame_ = data_frame[
        ~(data_frame[[weight, value]].isna().any(axis=1))
    ]
    mean = weighted_mean(data_frame_, value=value, weight=weight)
    return (data_frame_[weight] * ((data_frame_[value] - mean) ** 2)).sum()

def weighted_skewness(data_frame, value, weight):
    """Calculates skewness from values and weights.
    
    Parameters
    ----------
    data_frame:
        The table containing weights and values.
    value:
        Column containing values.
    weight:
        Column containing weights corresponding to values.

    Returns
    -------
    result
        The resulting skewness defined as:
            sum(weights * ((values - mean) ** 3)) / (sum(weights * ((values - mean) ** 2)) ** (3/2))
    """
    data_frame_ = data_frame[
        ~(data_frame[[weight, value]].isna().any(axis=1))
    ]
    mean = weighted_mean(data_frame_, value=value, weight=weight)
    variance = weighted_variance(data_frame_, value=value, weight=weight)
    return (data_frame_[weight] * ((data_frame_[value] - mean) ** 3)).sum() / (variance ** (3/2))
