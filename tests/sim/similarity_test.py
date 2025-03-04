from numpy import allclose, array, empty, float64, inner, zeros
from numpy.linalg import norm
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import raises

from pymmunomics.helper.exception import NotImplementedError
from pymmunomics.sim.similarity import (
    binding_similarity,
    make_similarity,
)

class TestBindingSimilarity:
    def test_both_genes_differ(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar2", "baz2"]),
        )
        expected = 0.0
        assert actual == expected

    def test_first_gene_differs(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar2", "baz"]),
        )
        expected = 0.0
        assert actual == expected

    def test_second_gene_differs(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar", "baz2"]),
        )
        expected = 0.0
        assert actual == expected

    def test_both_genes_same(self):
        actual = binding_similarity(
            array(["fofo", "bar", "baz"]),
            array(["ofof", "bar", "baz"]),
        )
        expected = 0.3**2
        assert actual == expected

    def test_identical(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar", "baz"]),
        )
        expected = 1.0
        assert actual == expected

class TestSimilarityFromDataFrame:
    def test_symmetric_matrix_multiple_communities(self):
        similarity = make_similarity(
            similarity=DataFrame(
                columns=["species1", "species2", "species3"],
                index=["species1", "species2", "species3"],
                data=[
                    [1.0, 0.5, 0.2],
                    [0.5, 1.0, 0.1],
                    [0.2, 0.1, 1.0],
                ],
            ),
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.75, 0.44, 0.  , 0.2 ],
            [0.75, 0.22, 0.  , 0.1 ],
            [0.15, 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_symmetric_matrix_single_community(self):
        similarity = make_similarity(
            similarity=DataFrame(
                columns=["species1", "species2", "species3"],
                index=["species1", "species2", "species3"],
                data=[
                    [1.0, 0.5, 0.2],
                    [0.5, 1.0, 0.1],
                    [0.2, 0.1, 1.0],
                ],
            ),
        )
        species_frequencies = array([
            [0.5],
            [0.5],
            [0.0],
        ])
        expected = array([
            [0.75],
            [0.75],
            [0.15],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_asymmetric_square_matrix_multiple_communities(self):
        similarity = make_similarity(
            similarity=DataFrame(
                columns=["species1", "species2", "species3"],
                index=["species1", "species2", "species3"],
                data=[
                    [0.5, 1.0, 0.1],
                    [1.0, 0.5, 0.2],
                    [0.2, 0.1, 1.0],
                ],
            ),
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.75, 0.22, 0.  , 0.1 ],
            [0.75, 0.44, 0.  , 0.2 ],
            [0.15, 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_asymmetric_square_matrix_single_community(self):
        similarity = make_similarity(
            similarity=DataFrame(
                columns=["species1", "species2", "species3"],
                index=["species1", "species2", "species3"],
                data=[
                    [0.5, 1.0, 0.1],
                    [1.0, 0.5, 0.2],
                    [0.2, 0.1, 1.0],
                ],
            ),
        )
        species_frequencies = array([
            [0.3],
            [0.0],
            [0.7],
        ])
        expected = array([
            [0.22],
            [0.44],
            [0.76],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_rectangular_matrix_multiple_communities(self):
        similarity = make_similarity(
            similarity=DataFrame(
                columns=["species1", "species2"],
                index=["species1", "species2", "species3"],
                data=[
                    [1.0, 0.2],
                    [0.5, 0.1],
                    [0.2, 1.0],
                ],
            ),
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.5 , 0.44, 0.  , 0.2 ],
            [0.25, 0.22, 0.  , 0.1 ],
            [0.1 , 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )

    def test_rectangular_matrix_single_community(self):
        similarity = make_similarity(
            similarity=DataFrame(
                columns=["species1", "species2"],
                index=["species1", "species2", "species3"],
                data=[
                    [1.0, 0.2],
                    [0.5, 0.1],
                    [0.2, 1.0],
                ],
            ),
        )
        species_frequencies = array([
            [0.5],
            [0.0],
        ])
        expected = array([
            [0.5 ],
            [0.25],
            [0.1 ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

class TestSimilarityFromArray:
    def test_symmetric_matrix_multiple_communities(self):
        similarity = make_similarity(
            similarity=array([
                [1.0, 0.5, 0.2],
                [0.5, 1.0, 0.1],
                [0.2, 0.1, 1.0],
            ]),
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.75, 0.44, 0.  , 0.2 ],
            [0.75, 0.22, 0.  , 0.1 ],
            [0.15, 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_symmetric_matrix_single_community(self):
        similarity = make_similarity(
            similarity=array([
                [1.0, 0.5, 0.2],
                [0.5, 1.0, 0.1],
                [0.2, 0.1, 1.0],
            ]),
        )
        species_frequencies = array([
            [0.5],
            [0.5],
            [0.0],
        ])
        expected = array([
            [0.75],
            [0.75],
            [0.15],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_asymmetric_square_matrix_multiple_communities(self):
        similarity = make_similarity(
            similarity=array([
                [0.5, 1.0, 0.1],
                [1.0, 0.5, 0.2],
                [0.2, 0.1, 1.0],
            ]),
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.75, 0.22, 0.  , 0.1 ],
            [0.75, 0.44, 0.  , 0.2 ],
            [0.15, 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_asymmetric_square_matrix_single_community(self):
        similarity = make_similarity(
            similarity=array([
                [0.5, 1.0, 0.1],
                [1.0, 0.5, 0.2],
                [0.2, 0.1, 1.0],
            ]),
        )
        species_frequencies = array([
            [0.3],
            [0.0],
            [0.7],
        ])
        expected = array([
            [0.22],
            [0.44],
            [0.76],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_rectangular_matrix_multiple_communities(self):
        similarity = make_similarity(
            similarity=array([
                [1.0, 0.2],
                [0.5, 0.1],
                [0.2, 1.0],
            ]),
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.5 , 0.44, 0.  , 0.2 ],
            [0.25, 0.22, 0.  , 0.1 ],
            [0.1 , 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )

    def test_rectangular_matrix_single_community(self):
        similarity = make_similarity(
            similarity=array([
                [1.0, 0.2],
                [0.5, 0.1],
                [0.2, 1.0],
            ]),
        )
        species_frequencies = array([
            [0.5],
            [0.0],
        ])
        expected = array([
            [0.5 ],
            [0.25],
            [0.1 ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

class TestSimilarityFromFile:
    def test_symmetric_matrix_multiple_communities(self, tmp_path):
        filecontent = (
            "species1\tspecies2\tspecies3\n"
            "1.0\t0.5\t0.2\n"
            "0.5\t1.0\t0.1\n"
            "0.2\t0.1\t1.0\n"
        )
        filepath = f"{tmp_path}/sim.tsv"
        with open(filepath, "w") as file:
            file.write(filecontent)
        similarity = make_similarity(
            similarity=filepath,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.75, 0.44, 0.  , 0.2 ],
            [0.75, 0.22, 0.  , 0.1 ],
            [0.15, 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_symmetric_matrix_single_community(self, tmp_path):
        filecontent = (
            "species1\tspecies2\tspecies3\n"
            "1.0\t0.5\t0.2\n"
            "0.5\t1.0\t0.1\n"
            "0.2\t0.1\t1.0\n"
        )
        filepath = f"{tmp_path}/sim.tsv"
        with open(filepath, "w") as file:
            file.write(filecontent)
        similarity = make_similarity(
            similarity=filepath,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5],
            [0.5],
            [0.0],
        ])
        expected = array([
            [0.75],
            [0.75],
            [0.15],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_asymmetric_square_matrix_multiple_communities(self, tmp_path):
        filecontent = (
            "species1\tspecies2\tspecies3\n"
            "0.5\t1.0\t0.1\n"
            "1.0\t0.5\t0.2\n"
            "0.2\t0.1\t1.0\n"
        )
        filepath = f"{tmp_path}/sim.tsv"
        with open(filepath, "w") as file:
            file.write(filecontent)
        similarity = make_similarity(
            similarity=filepath,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.75, 0.22, 0.  , 0.1 ],
            [0.75, 0.44, 0.  , 0.2 ],
            [0.15, 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_asymmetric_square_matrix_single_community(self, tmp_path):
        filecontent = (
            "species1\tspecies2\tspecies3\n"
            "0.5\t1.0\t0.1\n"
            "1.0\t0.5\t0.2\n"
            "0.2\t0.1\t1.0\n"
        )
        filepath = f"{tmp_path}/sim.tsv"
        with open(filepath, "w") as file:
            file.write(filecontent)
        similarity = make_similarity(
            similarity=filepath,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.3],
            [0.0],
            [0.7],
        ])
        expected = array([
            [0.22],
            [0.44],
            [0.76],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_rectangular_matrix_multiple_communities(self, tmp_path):
        filecontent = (
            "species1\tspecies2\n"
            "1.0\t0.2\n"
            "0.5\t0.1\n"
            "0.2\t1.0\n"
        )
        filepath = f"{tmp_path}/sim.tsv"
        with open(filepath, "w") as file:
            file.write(filecontent)
        similarity = make_similarity(
            similarity=filepath,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.5 , 0.44, 0.  , 0.2 ],
            [0.25, 0.22, 0.  , 0.1 ],
            [0.1 , 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )

    def test_rectangular_matrix_single_community(self, tmp_path):
        filecontent = (
            "species1\tspecies2\n"
            "1.0\t0.2\n"
            "0.5\t0.1\n"
            "0.2\t1.0\n"
        )
        filepath = f"{tmp_path}/sim.tsv"
        with open(filepath, "w") as file:
            file.write(filecontent)
        similarity = make_similarity(
            similarity=filepath,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5],
            [0.0],
        ])
        expected = array([
            [0.5 ],
            [0.25],
            [0.1 ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        assert allclose(actual, expected)

    def test_chunk_size_larger_than_matrix(self, tmp_path):
        filecontent = (
            "species1\tspecies2\n"
            "1.0\t0.2\n"
            "0.5\t0.1\n"
            "0.2\t1.0\n"
        )
        filepath = f"{tmp_path}/sim.tsv"
        with open(filepath, "w") as file:
            file.write(filecontent)
        similarity = make_similarity(
            similarity=filepath,
            chunk_size=5,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        expected = array([
            [0.5 , 0.44, 0.  , 0.2 ],
            [0.25, 0.22, 0.  , 0.1 ],
            [0.1 , 0.76, 0.  , 1.  ],
        ])
        actual = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )

class TestSimilarityFromFunction:
    def test_symmetric_matrix_multiple_communities(self):
        X = array([
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, -3.0],
        ])
        similarity = make_similarity(
            similarity=inner,
            X=X,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        actual_weighted_similarities = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        expected_weighted_similarities = array([
            [ 1. , -0.1,  0. , -1. ],
            [ 0. ,  0. ,  0. ,  0. ],
            [-0.5,  8.8,  0. , 13. ],
        ])
        assert allclose(
            actual_weighted_similarities,
            expected_weighted_similarities,
        )

    def test_symmetric_matrix_multiple_communities_store_matrix(self):
        X = array([
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, -3.0],
        ])
        actual_similarities = zeros(shape=(3,3), dtype=float64)
        similarity = make_similarity(
            similarity=inner,
            X=X,
            # similarities_out=actual_similarities,  TODO: add this back in
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        actual_weighted_similarities = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        expected_similarities = array([
            [ 2.,  0., -1.],
            [ 0.,  0.,  0.],
            [-1.,  0., 13.],
        ])
        expected_weighted_similarities = array([
            [ 1. , -0.1,  0. , -1. ],
            [ 0. ,  0. ,  0. ,  0. ],
            [-0.5,  8.8,  0. , 13. ],
        ])
        # assert allclose(actual_similarities, expected_similarities) TODO: add this back in
        assert allclose(
            actual_weighted_similarities,
            expected_weighted_similarities,
        )

    def test_symmetric_matrix_single_community(self):
        X = array([
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, -3.0],
        ])
        similarity = make_similarity(
            similarity=inner,
            X=X,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5],
            [0.5],
            [0.0],
        ])
        actual_weighted_similarities = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        expected_weighted_similarities = array([
            [ 1. ],
            [ 0. ],
            [-0.5],
        ])
        assert allclose(
            actual_weighted_similarities,
            expected_weighted_similarities,
        )

    def test_rectangular_matrix_multiple_communities(self):
        X = array([
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, -3.0],
        ])
        Y = array([
            [0.5, 0.5],
            [0.0, -2.0],
        ])
        similarity = make_similarity(
            similarity=inner,
            X=X,
            Y=Y,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        actual_weighted_similarities = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        expected_weighted_similarities = array([
            [ 0.5 , -1.1 ,  0.  , -2.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [-0.25,  4.05,  0.  ,  6.  ],
        ])
        assert allclose(
            actual_weighted_similarities,
            expected_weighted_similarities,
        )

    def test_rectangular_matrix_multiple_communities_store_matrix(self):
        X = array([
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, -3.0],
        ])
        Y = array([
            [0.5, 0.5],
            [0.0, -2.0],
        ])
        actual_similarities = zeros(shape=(3,2), dtype=float64)
        similarity = make_similarity(
            similarity=inner,
            X=X,
            Y=Y,
            # similarities_out=actual_similarities, TODO: add this back in
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5, 0.3, 0.0, 0.0],
            [0.0, 0.7, 0.0, 1.0],
        ])
        actual_weighted_similarities = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        expected_similarities = array([
            [ 1. , -2. ],
            [ 0. ,  0. ],
            [-0.5,  6. ],
        ])
        expected_weighted_similarities = array([
            [ 0.5 , -1.1 ,  0.  , -2.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [-0.25,  4.05,  0.  ,  6.  ],
        ])
        # assert allclose(actual_similarities, expected_similarities) TODO: add this back in
        assert allclose(
            actual_weighted_similarities,
            expected_weighted_similarities,
        )

    def test_rectangular_matrix_single_community(self):
        X = array([
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, -3.0],
        ])
        Y = array([
            [0.5, 0.5],
            [0.0, -2.0],
        ])
        similarity = make_similarity(
            similarity=inner,
            X=X,
            Y=Y,
            chunk_size=2,
        )
        species_frequencies = array([
            [0.5],
            [0.0],
        ])
        actual_weighted_similarities = similarity.weighted_similarities(
            species_frequencies=species_frequencies,
        )
        expected_weighted_similarities = array([
            [ 0.5 ],
            [ 0.  ],
            [-0.25],
        ])
        assert allclose(
            actual_weighted_similarities,
            expected_weighted_similarities,
        )

class TestMakeSimilarity:
    def test_similarity_not_implemented(self):
        with raises(NotImplementedError):
            make_similarity(similarity=1)
