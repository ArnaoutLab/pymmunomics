from pytest import fixture, mark, raises
from numpy import (
    allclose,
    arange,
    float64,
    full,
    memmap,
    ndarray,
    ones,
    zeros,
)
from pymmunomics.helper.numpy_helpers import (
    apply_to_memmapped,
    MemMapSpec,
)

def add_arrays(x, y, const=None):
    if const is None:
        return x + y
    else:
        return x + y + const

def write_data(arr, data):
    assert arr.shape == data.shape, "Test logic error: inconsistent array shapes"
    arr[:, :] = data

@fixture
def x_filepath(tmp_path):
    return f"{tmp_path}/x.npy"

@fixture
def y_filepath(tmp_path):
    return f"{tmp_path}/y.npy"

def write_memmap(filepath, data, memmap_spec):
    assert memmap_spec.shape == data.shape, "Test logic error: inconsistent array shapes"
    arr = memmap(
        filepath,
        dtype=memmap_spec.dtype,
        mode="w+",
        offset=memmap_spec.offset,
        shape=memmap_spec.shape,
        order=memmap_spec.order,
    )
    arr[:,:] = data
    arr._mmap.close()

def test_addition(x_filepath, y_filepath):
    x_spec = MemMapSpec(dtype=float64, mode="r+", shape=(3, 3))
    y_spec = MemMapSpec(dtype=float64, mode="r+", shape=(3, 3))
    write_memmap(
        filepath=x_filepath,
        data=ones((3, 3)),
        memmap_spec=x_spec,
    )
    write_memmap(
        filepath=y_filepath,
        data=2 * ones((3, 3)),
        memmap_spec=y_spec,
    )

    arg_to_filepath_memmap_spec = {
        "x": (x_filepath, x_spec),
        "y": (y_filepath, y_spec),
    }
    func_kwargs = {}
    expected = 3 * ones((3, 3))
    actual = apply_to_memmapped(
        func=add_arrays,
        arg_to_filepath_memmap_spec=arg_to_filepath_memmap_spec,
        func_kwargs=func_kwargs,
    )
    assert allclose(actual, expected)

def test_func_kwargs(x_filepath, y_filepath):
    x_spec = MemMapSpec(dtype=float64, mode="r+", shape=(3, 3))
    y_spec = MemMapSpec(dtype=float64, mode="r+", shape=(3, 3))
    write_memmap(
        filepath=x_filepath,
        data=ones((3, 3)),
        memmap_spec=x_spec,
    )
    write_memmap(
        filepath=y_filepath,
        data=2 * ones((3, 3)),
        memmap_spec=y_spec,
    )

    arg_to_filepath_memmap_spec = {
        "x": (x_filepath, x_spec),
        "y": (y_filepath, y_spec),
    }
    func_kwargs = {"const": 3}
    expected = 6 * ones((3, 3))
    actual = apply_to_memmapped(
        func=add_arrays,
        arg_to_filepath_memmap_spec=arg_to_filepath_memmap_spec,
        func_kwargs=func_kwargs,
    )
    assert allclose(actual, expected)

def test_write_new_memmap(x_filepath):
    x_spec = MemMapSpec(dtype=float64, mode="w+", shape=(3, 3))

    arg_to_filepath_memmap_spec = {
        "arr": (x_filepath, x_spec),
    }
    func_kwargs = {"data": 7 * ones((3, 3))}
    expected = 7 * ones((3, 3))
    apply_to_memmapped(
        func=write_data,
        arg_to_filepath_memmap_spec=arg_to_filepath_memmap_spec,
        func_kwargs=func_kwargs,
    )
    actual = memmap(
        x_filepath,
        dtype=x_spec.dtype,
        mode="r+",
        shape=x_spec.shape,
    )
    assert allclose(actual, expected)
    actual._mmap.close()
