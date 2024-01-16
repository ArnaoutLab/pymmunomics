from os import mkdir
from pathlib import Path

from pytest import raises

from pymmunomics.helper.exception import InvalidArgumentError
from pymmunomics.helper.generic_helpers import (
    chain_update,
    glob_files,
    map_has_substring,
    map_replace,
    Pipeline,
    set_intersections,
    set_unions,
)

class TestChainUpdate:
    def test_no_duplicates_no_transitives(self):
        mappings = [
            {"a": "b"},
            {"c": "d", "e": "f"},
            {1: 100},
        ]
        expected_result = {
            "a": "b",
            "c": "d", "e": "f",
            1: 100,
        }
        actual_result = chain_update(mappings=mappings)
        assert actual_result == expected_result

    def test_duplicatest_no_transitives(self):
        mappings = [
            {"a": "b"},
            {"a": "x", "c": "d", "e": "f"},
            {1: 100, "a": "y", "e": "z"},
        ]
        expected_result = {
            "a": "y",
            "c": "d", "e": "z",
            1: 100,
        }
        actual_result = chain_update(mappings=mappings)
        assert actual_result == expected_result

    def test_transitives_no_duplicates(self):
        mappings = [
            {"a": "b"},
            {"c": "d", "e": "f", "b": "foo"},
            {1: 100, "f": "bazinga", "foo": "zap"},
        ]
        expected_result = {
            "a": "zap",
            "c": "d", "e": "bazinga", "b": "zap",
            1: 100, "f": "bazinga", "foo": "zap",
        }
        actual_result = chain_update(mappings=mappings)
        assert actual_result == expected_result

    def test_transitives_duplicates(self):
        mappings = [
            {"a": "b", "x": "b", "u": "v"},
            {"c": "d", "e": "f", "b": "foo", "u": "w", "t": "s"},
            {1: 100, "f": "bazinga", "foo": "zap", "x": "y", "u": "p", "t": "q"},
        ]
        expected_result = {
            "a": "zap", "x": "y", "u": "p",
            "c": "d", "e": "bazinga", "b": "zap", "t": "q",
            1: 100, "f": "bazinga", "foo": "zap",
        }
        actual_result = chain_update(mappings=mappings)
        assert actual_result == expected_result

    def test_empty_input(self):
        mappings = []
        assert chain_update(mappings) == {}

class TestGlobFiles:
    def test_nonnested_glob(self, tmp_path):
        pathname = f"{tmp_path}/*"

        Path(f"{tmp_path}/file1").touch()
        Path(f"{tmp_path}/file2.ext").touch()

        expected_filepaths = {f"{tmp_path}/file1", f"{tmp_path}/file2.ext"}
        actual_filepaths = set(glob_files(pathname=pathname))

        assert actual_filepaths == expected_filepaths

    def test_nested_glob(self, tmp_path):
        pathname = f"{tmp_path}/**"

        Path(f"{tmp_path}/file1").touch()
        Path(f"{tmp_path}/file2.ext").touch()
        mkdir(f"{tmp_path}/directory")
        Path(f"{tmp_path}/directory/file1").touch()

        expected_filepaths = {
            f"{tmp_path}/file1",
            f"{tmp_path}/file2.ext",
            f"{tmp_path}/directory/file1",
        }
        actual_filepaths = set(glob_files(pathname=pathname, recursive=True))

        assert actual_filepaths == expected_filepaths

class TestMapHasSubstring:
    def test_infix_match(self):
        s = "foobarbaz"
        substrings = ["zinga", "bar", "zoo"]
        has_substring = map_has_substring(s=s, substrings=substrings)
        assert has_substring

    def test_prefix_match(self):
        s = "foobarbaz"
        substrings = ["zinga", "foo", "zoo"]
        has_substring = map_has_substring(s=s, substrings=substrings)
        assert has_substring

    def test_suffix_match(self):
        s = "foobarbaz"
        substrings = ["zinga", "baz", "zoo"]
        has_substring = map_has_substring(s=s, substrings=substrings)
        assert has_substring

    def test_no_match(self):
        s = "foobarbaz"
        substrings = ["bazinga", "fooo", "foobaz"]
        has_substring = map_has_substring(s=s, substrings=substrings)
        assert not has_substring

    def test_empty_substrings(self):
        s = "foobarbaz"
        substrings = []
        has_substring = map_has_substring(s=s, substrings=substrings)
        assert not has_substring

class TestMapReplace:
    def test_single_infix_match(self):
        s = "foobarbaz"
        infix_map = {"bar": "ZAP", "fooo": "BAZINGA"}
        mode = "infix"
        expected_result = "fooZAPbaz"
        actual_result = map_replace(s=s, infix_map=infix_map, mode=mode)
        assert actual_result == expected_result

    def test_multiple_infix_match(self):
        s = "afoobaarbaza"
        infix_map = {"a": "ZAP", "bazinga": "BAZINGA"}
        mode = "infix"
        expected_result = "ZAPfoobZAPZAPrbZAPzZAP"
        actual_result = map_replace(s=s, infix_map=infix_map, mode=mode)
        assert actual_result == expected_result

    def test_no_infix_match(self):
        s = "foobarbaz"
        infix_map = {"baar": "ZAP", "bazinga": "BAZINGA"}
        mode = "infix"
        expected_result = "foobarbaz"
        actual_result = map_replace(s=s, infix_map=infix_map, mode=mode)
        assert actual_result == expected_result

    def test_prefix_match(self):
        s = "foofoobarbaz"
        infix_map = {"foo": "ZAP", "baz": "BAZINGA"}
        mode = "prefix"
        expected_result = "ZAPfoobarbaz"
        actual_result = map_replace(s=s, infix_map=infix_map, mode=mode)
        assert actual_result == expected_result

    def test_no_prefix_match(self):
        s = "foobarbaz"
        infix_map = {"bar": "ZAP", "baz": "BAZINGA"}
        mode = "prefix"
        expected_result = "foobarbaz"
        actual_result = map_replace(s=s, infix_map=infix_map, mode=mode)
        assert actual_result == expected_result

    def test_suffix_match(self):
        s = "foobarbazbaz"
        infix_map = {"baz": "ZAP", "foo": "BAZINGA"}
        mode = "suffix"
        expected_result = "foobarbazZAP"
        actual_result = map_replace(s=s, infix_map=infix_map, mode=mode)
        assert actual_result == expected_result

    def test_no_suffix_match(self):
        s = "foobarbaz"
        infix_map = {"bar": "ZAP", "foo": "BAZINGA"}
        mode = "suffix"
        expected_result = "foobarbaz"
        actual_result = map_replace(s=s, infix_map=infix_map, mode=mode)
        assert actual_result == expected_result

class TestPipeline:
    def fake_function_1(*args, **kwargs):
        if "add" in kwargs:
            add = kwargs["add"] + sum(args)
        else:
            add = 1 + sum(args)
        return tuple([arg + add for arg in args])
        
    def fake_function_2(*args, **kwargs):
        return str((args, kwargs))

    def test_single_function(self):
        steps = [TestPipeline.fake_function_1]
        pipeline = Pipeline(steps=steps)
        x = 10
        args = ()
        kwargs = {}

        expected_result = (21,)
        actual_result = pipeline(x, *args, **kwargs,)
        assert actual_result == expected_result

    def test_multiple_functions(self):
        steps = [
            TestPipeline.fake_function_1,
            TestPipeline.fake_function_2,
        ]
        pipeline = Pipeline(steps=steps)
        x = 10
        args = ()
        kwargs = {}

        expected_result = str((((21,),), {}))
        actual_result = pipeline(x, *args, **kwargs,)
        assert actual_result == expected_result

    def test_args_kwargs(self):
        steps = [
            TestPipeline.fake_function_1,
            TestPipeline.fake_function_2,
        ]
        pipeline = Pipeline(steps=steps)
        x = 10
        args = (100,)
        kwargs = {"add": 1000}

        expected_result = str((((1120, 1210),), {}))
        actual_result = pipeline(x, *args, **kwargs,)
        assert actual_result == expected_result

    def test_step_kwargs(self):
        steps = [
            TestPipeline.fake_function_1,
            TestPipeline.fake_function_2,
        ]
        pipeline = Pipeline(steps=steps)
        x = 10
        args = ()
        kwargs = {}
        step_kwargs = [{"add": 1000}, {"foo": 15}]

        expected_result = str((((1020,),), {"foo": 15}))
        actual_result = pipeline(x, *args, step_kwargs=step_kwargs, **kwargs)
        assert actual_result == expected_result

    def test_args_kwargs_kwargs(self):
        steps = [
            TestPipeline.fake_function_1,
            TestPipeline.fake_function_2,
        ]
        pipeline = Pipeline(steps=steps)
        x = 10
        args = (100,)
        kwargs = {"add": 1000}
        step_kwargs = [{}, {"foo": 15}]

        expected_result = str((((1120,1210),), {"foo": 15}))
        actual_result = pipeline(x, *args, step_kwargs=step_kwargs, **kwargs)
        assert actual_result == expected_result

    def test_invalid_argument(self):
        steps = [
            TestPipeline.fake_function_1,
            TestPipeline.fake_function_2,
        ]
        pipeline = Pipeline(steps=steps)
        x = 10
        step_kwargs = [{"add": 1000}, {"foo": 15}]

        with raises(InvalidArgumentError):
            pipeline(x, step_kwargs=step_kwargs[:1])

        pipeline = Pipeline(steps=steps[:1])
        x = 10
        step_kwargs = [{"add": 1000}, {"foo": 15}]

        with raises(InvalidArgumentError):
            pipeline(x, step_kwargs=step_kwargs)

        with raises(InvalidArgumentError):
            Pipeline(steps=[])


class TestSetIntersections:
    def test_multiple_intersecting_sets(self):
        sets = [{"a", "b", "c"}, {"a", "c"}, {"a", "d"}]
        expected_intersection = {"a"}
        actual_intersection = set_intersections(sets=sets)
        assert actual_intersection == expected_intersection

    def test_multiple_sets_empty_intersection(self):
        sets = [{"a", "b", "c"}, {"a", "c"}, {"b", "d"}]
        expected_intersection = set()
        actual_intersection = set_intersections(sets=sets)
        assert actual_intersection == expected_intersection

    def test_single_set(self):
        sets = [{"a", "b", "c"}]
        expected_intersection = {"a", "b", "c"}
        actual_intersection = set_intersections(sets=sets)
        assert actual_intersection == expected_intersection

    def test_no_sets(self):
        sets = []
        expected_intersection = set()
        actual_intersection = set_intersections(sets=sets)
        assert actual_intersection == expected_intersection

class TestSetUnions:
    def test_multiple_overlapping_sets(self):
        sets = [{"a", "b", "c"}, {"a", "c"}, {"a", "d"}]
        expected_union = {"a", "b", "c", "d"}
        actual_union = set_unions(sets=sets)
        assert actual_union == expected_union

    def test_multiple_disjoint_sets(self):
        sets = [{"a", "b", "c"}, {"d", "e"}, {"f", "g"}]
        expected_union = {"a", "b", "c", "d", "e", "f", "g"}
        actual_union = set_unions(sets=sets)
        assert actual_union == expected_union

    def test_single_set(self):
        sets = [{"a", "b", "c"}]
        expected_union = {"a", "b", "c"}
        actual_union = set_unions(sets=sets)
        assert actual_union == expected_union

    def test_no_sets(self):
        sets = []
        expected_union = set()
        actual_union = set_unions(sets=sets)
        assert actual_union == expected_union
