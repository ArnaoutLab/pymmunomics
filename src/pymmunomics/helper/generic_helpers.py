from contextlib import contextmanager
from copy import deepcopy
from glob import glob
from itertools import repeat
from os import remove
from os.path import isfile
from shutil import copyfileobj
from sys import stdout
from typing import Any, Callable, Hashable, Iterable, Literal, Sequence, Union

from pymmunomics.helper.exception import InvalidArgumentError

def call_method(
    obj: Any,
    method: str,
    *args,
    **kwargs,
):
    return getattr(obj, method)(*args, **kwargs)

def chain_update(
    mappings: Sequence[dict],
) -> dict:
    """Combines dictionaries transitively.

    Parameters
    ----------
    mappings:
        The dictionaries to combine in order of transitivity and
        precedence.

    Returns
    -------
    chained_dict:
        A dictionary combining all keys and successively replacing
        mapped values if preceding values are keys in succeeding
        dictionaries. Succeeding dictionary key-value pairs take
        precedence and overwrite preceeding key-value pairs before
        transitive replacements.
    """
    if len(mappings) == 0:
        return {}
    else:
        result = deepcopy(mappings[0])
        for mapping in mappings[1:]:
            # Update duplicate mappings
            for key in result:
                if key in mapping:
                    result[key] = mapping[key]
            # Transitive mappings
            transitives = {}
            for key in result:
                if result[key] in mapping:
                    transitives[key] = mapping[result[key]]
            result.update(transitives)
            # missing mappings
            for key in mapping:
                if key not in result:
                    result[key] = mapping[key]
            # result.update(mapping)
            # for key, value in result.items():
            #     if value in mapping:
            #         result[key] = mapping[value]
    return result

# def concatenate_files(
#     input_filepaths: Sequence[str],
#     output_filepath: str,
#     read_mode: str = "r",
#     write_mode: str = "w",
#     header: Union[int, None] = None,
#     delete_input_files: bool = False,
# ):
#     """Concatenates data files.

#     Parameters
#     ----------
#     input_filepaths:
#         Path to files to concatenate.
#     output_filepath:
#         Path to file in which to store concatenation of input files.
#     read_mode:
#         Input file open mode.
#     write_mode:
#         Output file open mode.
#     header:
#         Number of rows that are considered header rows. The header of
#         the first input file is used in the destination file and skipped
#         in all remaining input files.
#     delete_input_files:
#         Determines whether or not input files are deleted after
#         concatenating.
#     """
#     with open(output_filepath, write_mode) as output_file:
#         with open(input_filepaths[0], read_mode) as input_file:
#             copyfileobj(input_file, output_file)
#         for input_filepath in input_filepaths[1:]:
#             with open(input_filepath, read_mode) as input_file:
#                 if header is not None:
#                     for _ in range(header):
#                         input_file.readline()
#                 copyfileobj(input_file, output_file)
#     if delete_input_files:
#         for filepath in input_filepaths:
#             remove(filepath)
#     return None

def glob_files(pathname: str, *args, **kwargs):
    """Returns matches that are files.

    Parameters
    ----------
    pathname:
        See official documentation of builtin function ``glob.glob``.
    args, kwargs:
        Passed to glob.glob. See official documentation.

    Returns
    -------
    All files matched by `glob.glob` (excluding directories).
    """
    all_paths = glob(pathname, *args, **kwargs)
    filepaths = [p for p in all_paths if isfile(p)]
    return filepaths

def map_has_substring(s: str, substrings: Iterable[str]) -> bool:
    """Determines if s has any of the substrings.

    Parameters
    ----------
    s:
        String to search for substrings.
    substrings:
        Substrings to search for.

    Returns
    -------
    has_substring:
        Indicates whether or not s has one of the substrings.
    """
    return any(
        substring in s
        for substring
        in substrings
    )

def map_replace(
    s: str,
    infix_map: dict[str, str],
    mode: Literal["infix", "prefix", "suffix"] = "infix",
):
    """Replaces infix of string if infix found.

    Parameters
    ----------
    s:
        String to check for infix replacement.
    infix_map:
        Maps infixes that need to be replaced to their replacement.
    mode:
        "infix" leads to all occurrences of an infix to be replaced.
        "prefix" and "suffix" leads to only replacing prefix or suffix
        of strings if prefix or suffix are in `infix_map`, respectively.

    Returns
    -------
    new_s:
        Same as input string if it has none of the infixes
        (prefices/suffices). Else, the first found infix (prefix/suffix)
        from `infix_map` - in no guaranteed order - is replaced (all
        occurrences (prefix/suffix only)).
    """
    for infix, new_infix in infix_map.items():
        if mode == "infix" and infix in s:
            return s.replace(infix, new_infix)
        elif mode == "prefix" and s.startswith(infix):
            return f"{new_infix}{s[len(infix):]}"
        elif mode == "suffix" and s.endswith(infix):
            return f"{s[:-len(infix)]}{new_infix}"
    else:
        return s

# @contextmanager
# def open_file_or_stdout(
#     filepath: Union[str, None] = None,
#     **open_kwargs,
# ):
#     """Context manager deciding if file or ``sys.stdout`` is returned.

#     Parameters
#     ----------
#     filepath:
#         If None, ``sys.stdout`` is used. If str, builting ``open``
#         function is used.
#     open_kwargs:
#         Keyword arguments for builtin ``open``. If using ``sys.stdout``,
#         these keyword arguments are ignored.
#     """
#     if filepath is None:
#         try:
#             yield stdout
#         finally:
#             pass
#     else:
#         try:
#             with open(filepath, **open_kwargs) as file_handle:
#                 yield file_handle
#         except Exception:
#             raise
#         finally:
#             pass

# def partition_range(range_: range, num_chunks: int):
#     """Splits range_ into evenly sized consecutive subranges.

#     Parameters
#     ----------
#     range_:
#         The range to split into chunks.
#     num_chunks:
#         The number of subranges to split range_ into. Must be positive.

#     Returns
#     -------
#     A list of range objects. The ranges are evenly sized. When range_
#     doesn't divide evenly by num_chunks, the ranges near the end of the
#     returned list are 1 larger than ranges near the beginning.

#     Raises
#     ------
#     InvalidArgumentError when num_chunks is not positive.
#     """
#     if num_chunks <= 0:
#         raise InvalidArgumentError(
#             f"Range can not be split into a non-positive number of chunks."
#         )
#     small_chunk_size, num_big_chunks = divmod(len(range_), num_chunks)
#     num_small_chunks = num_chunks - num_big_chunks
#     small_chunks_start = range_.start
#     small_chunks = [
#         range(
#             small_chunks_start + (i * small_chunk_size),
#             small_chunks_start + ((i + 1) * small_chunk_size),
#         )
#         for i in range(num_small_chunks)
#     ]
#     big_chunks_start = small_chunks_start + (num_small_chunks * small_chunk_size)
#     big_chunk_size = small_chunk_size + 1
#     big_chunks = [
#         range(
#             big_chunks_start + (i * big_chunk_size),
#             big_chunks_start + ((i + 1) * big_chunk_size),
#         )
#         for i in range(num_big_chunks)
#     ]
#     return [*small_chunks, *big_chunks]

class Pipeline:
    
    def __init__(self, steps: Sequence[Callable]):
        """Applies sequence of functions on successive results.
        
        Parameters
        ----------
        steps:
            The steps of the pipeline in order of execution. Each
            function is applied to its predecessor's return value.
        """
        if len(steps) < 1:
            raise InvalidArgumentError(
                "Pipeline must consist of at least one function."
            )
        self.steps = steps

    def __call__(
        self,
        *args,
        step_kwargs: Union[Sequence[dict], None] = None,
        **kwargs,
    ):
        """Applies pipeline to initial arguments.

        Parameters
        ----------
        step_kwargs:
            Items are passed as keyword arguments to corresponding
            pipeline steps. It must correspond in length with the number
            of steps in the pipeline.
        args, kwargs:
            The first function's positional and keyword arguments.

        Returns
        -------
        Return value of last member of `functions`.
        """
        if step_kwargs is None:
            step_kwargs = list(repeat({}, len(self.steps)))
        elif len(step_kwargs) != len(self.steps):
            raise InvalidArgumentError(
                "Must pass same number of function_kwargs as functions."
                " Instead, passed %s functions and %s function_kwargs",
                (len(self.steps), len(step_kwargs)),
            )
        result = self.steps[0](*args, **kwargs, **step_kwargs[0])
        for step, step_kwargs_ in zip(self.steps[1:], step_kwargs[1:]):
            result = step(result, **step_kwargs_)
        return result

def prepend(s: str, prefix: str):
    return f"{prefix}{s}"

def set_intersections(sets: Sequence[Iterable[Hashable]]):
    if len(sets) == 0:
        intersection = set()
    else:
        intersection = set(sets[0])
    for item in sets[1:]:
        intersection = intersection.intersection(set(item))
    return intersection

def set_unions(sets: Sequence[Iterable[Hashable]]):
    if len(sets) == 0:
        union = set()
    else:
        union = set(sets[0])
    for item in sets[1:]:
        union = union.union(set(item))
    return union
