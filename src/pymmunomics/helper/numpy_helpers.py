from typing import Callable, NamedTuple

from numpy import dtype, memmap

def apply_to_memmapped(
    func: Callable,
    arg_to_filepath_memmap_spec: dict[str, tuple],
    func_kwargs: dict,
):
    """Applies func to specified memory-mapped array arguments.

    If memory-mapped array file does not exist yet for one of the
    arguments, be sure to use mode "w+" in the corresponding MemMapSpec
    specification.

    Parameters
    ----------
    func:
        Function to apply to numpy.memmap arguments.
    arg_to_filepath_memmap_spec:
        Maps parameter names of func to pairs of str filepaths and
        ``MemMapSpec`` objects specifying location of the corresponding
        memory-mapped array (.npy) file and how to interpret them.
    func_kwargs: dict[str, Any]
        Additional keyword arguments for `func`.

    Returns
    -------
    The return value of `func` applied to the memory-mapped array
    arguments and keyword arguments.
    """
    arg_to_memmap = {}
    for arg, (filepath, spec) in arg_to_filepath_memmap_spec.items():
        arg_to_memmap[arg] = memmap(
            filepath,
            dtype=spec.dtype,
            mode=spec.mode,
            offset=spec.offset,
            shape=spec.shape,
            order=spec.order,
        )
    return func(**arg_to_memmap, **func_kwargs)

class MemMapSpec(NamedTuple):
    """Describes how to interpret a file as ``numpy.memmap``.

    See ``numpy.memmap`` for description of attributes.
    """

    dtype: type = dtype("B")
    mode: str = "r+"
    offset: int = 0
    shape: tuple = None
    order: str = "C"
