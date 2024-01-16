"""Module for calculating (weighted) similarities.

Classes
-------
Similarity
    Abstract base class for calculating (weighted) similarities.
SimilarityFromDataFrame
    Implements Similarity using a precomputed similarity matrix stored
    in a pandas.DataFrame.
SimilarityFromArray
    Implements Similarity using a precomputed similarity matrix stored
    in a numpy.ndarray or numpy.memmap.
SimilarityFromFile
    Implements Similarity by reading similarities from a csv or tsv 
    file in chunks.
SimilarityFromFunction:
    Implements Similarity by calculating pairwise similarities from a 
    user-specified callable.

Functions
---------
binding_similarity:
    Calculates binding similarities of antibodies or B-/T-cell
    receptors.
"""
from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union

from numpy import memmap, ndarray, empty, concatenate, float64
from numpy.typing import ArrayLike
from pandas import DataFrame, Index, MultiIndex, read_csv
from polyleven import levenshtein
from scipy.sparse import spmatrix
from ray import remote, get, put

from pymmunomics.helper.exception import NotImplementedError
from pymmunomics.helper.log import LOGGER

def binding_similarity(left: Sequence, right: Sequence):
    """
    Calculates binding similarities as a function of levenshtein
    distance via an empirically determined relationship between
    levenshtein distances and dissociation constant ratios.

    Parameters
    ----------
    left, right:
        Clonotypes to compare. Must have CDR3 sequence at its 0th
        coordinate, and V-/J-gene at 2nd and 3rd coordinates,
        respectively.

    Returns
    -------
    0 when V- or J-genes differ; otherwise, a function of the
    Levenshtein distance of the CDR3s learned empirically: `0.3 ** distance`
    """
    if (left[1], left[2]) != (right[1], right[2]):
        return 0.0
    else:
        return 0.3**levenshtein(left[0], right[0])

class Similarity(ABC):
    """Interface for classes computing (weighted) similarities."""

    def __init__(self, similarity: Union[DataFrame, ndarray, str, Callable]) -> None:
        """
        Parameters
        ----------
        similarity:
            A similarity matrix, a path to a similarity matrix, or a
            function that can be called on pairs of species to calculate
            similarities. Similarity matrix rows correspond to query
            species and columns to subject species.
        """
        self.similarity = similarity

    @abstractmethod
    def weighted_similarities(
        self, species_frequencies: Union[ndarray, spmatrix]
    ) -> ndarray:
        """Calculates weighted sums of similarities for each species.

        Parameters
        ----------
        species_frequencies:
            Contains the frequencies for each subject species (rows) in
            each subject (columns).

        Returns
        -------
        weighted_similarity_sums:
            A 2-d array of the same shape as `species_frequencies`,
            where rows correspond to query species and columns to
            subject species distributions, and each element is the
            expectation of similarity of the query species to subject
            species according to the corresponding frequency
            distribution of subject species.
        """
        pass


class SimilarityFromDataFrame(Similarity):
    """Implements Similarity using similarities stored in a data frame."""

    def weighted_similarities(
        self, species_frequencies: Union[ndarray, spmatrix]
    ) -> ndarray:
        return self.similarity.to_numpy() @ species_frequencies


class SimilarityFromArray(Similarity):
    """Implements Similarity using similarities stored in an array."""

    def weighted_similarities(
        self, species_frequencies: Union[ndarray, spmatrix]
    ) -> ndarray:
        return self.similarity @ species_frequencies


class SimilarityFromFile(Similarity):
    """Implements Similarity by using similarities stored in file.

    Similarity matrix rows are read from the file one chunk at a time.
    The size of chunks can be specified in numbers of rows to control
    memory load.
    """

    def __init__(self, similarity: str, chunk_size: int = 100) -> None:
        """
        Parameters
        ----------
        similarity:
            Path to a file containing a similarity matrix. The file
            should have a single-row header.
        chunk_size:
            Number of rows to read from similarity matrix at a time.
        """
        super().__init__(similarity=similarity)
        self.chunk_size = chunk_size
        self.n_similarity_rows = read_csv(
            self.similarity,
            sep=None,
            engine="python",
            dtype=float64,
            usecols=[0],
        ).shape[0]

    def weighted_similarities(
        self, species_frequencies: Union[ndarray, spmatrix]
    ) -> ndarray:
        weighted_similarities = empty(
            shape=(self.n_similarity_rows, species_frequencies.shape[1]),
            dtype=float64,
        )
        with read_csv(
            self.similarity,
            chunksize=self.chunk_size,
            sep=None,
            engine="python",
            dtype=float64,
        ) as similarity_matrix_chunks:
            i = 0
            for chunk in similarity_matrix_chunks:
                weighted_similarities[i : i + self.chunk_size, :] = (
                    chunk.to_numpy() @ species_frequencies
                )
                i += self.chunk_size
        return weighted_similarities


@remote
def weighted_similarity_chunk(
    similarity: Callable,
    X: ndarray,
    species_frequencies: ndarray,
    chunk_size: int,
    chunk_index: int,
    Y: ndarray = None,
) -> ndarray:
    if Y is None:
        Y = X
    chunk = X[chunk_index : chunk_index + chunk_size]
    similarities_chunk = empty(shape=(chunk.shape[0], Y.shape[0]))
    for i, row_i in enumerate(chunk):
        for j, row_j in enumerate(Y):
            similarities_chunk[i, j] = similarity(row_i, row_j)
    return (similarities_chunk @ species_frequencies, similarities_chunk)


class SimilarityFromFunction(Similarity):
    """Calculates similarities from user-specified similarity function."""

    def __init__(
        self,
        similarity: Callable,
        X: ndarray,
        Y: ndarray = None,
        similarities_out: ndarray = None,
        chunk_size: int = 100,
    ) -> None:
        """
        similarity:
            A Callable that calculates similarity between a pair of
            species. Must take two rows from X, or a row from X and one
            from Y as its arguments and return a numeric similarity
            value.
        X, Y:
            Each row contains the feature values for a given species.
            Rows in X correspond to rows in the calcluated similarity
            matrix (and therefore to rows in the weighted similarities).
            If `Y` is not specified, rows in X also correspond to
            columns in the calculated similarity matrix.
        similarities_out:
            If specified, similarity matrix will be stored in this
            array.
        chunk_size:
            Determines how many rows of the similarity matrix will be
            stored in memory at a time. In general, choosing a larger
            `chunk_size` will make the calculation faster, but will also
            require more memory.
        """
        super().__init__(similarity=similarity)
        self.X = X
        self.Y = Y
        self.similarities_out = similarities_out
        self.chunk_size = chunk_size

    def weighted_similarities(
        self, species_frequencies: Union[ndarray, spmatrix]
    ) -> ndarray:
        X_ref = put(self.X)
        Y_ref = put(self.Y)
        abundance_ref = put(species_frequencies)
        futures = []
        similarities_chunk_futures = []
        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            chunk_future = weighted_similarity_chunk.remote(
                similarity=self.similarity,
                X=X_ref,
                species_frequencies=abundance_ref,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
                Y=Y_ref,
            )
            futures.append(chunk_future)
        weighted_similarity_chunks = get(futures)
        if self.similarities_out is not None:
            for chunk_index, similarities_chunk in zip(
                range(0, self.X.shape[0], self.chunk_size),
                weighted_similarity_chunks,
            ):
                self.similarities_out[chunk_index:chunk_index+similarities_chunk[1].shape[0]] = similarities_chunk[1]
        return concatenate([
            similarities_chunk[0]
            for similarities_chunk
            in weighted_similarity_chunks
        ])

def make_similarity(
    similarity: Union[DataFrame, ndarray, str, Callable],
    X: ndarray = None,
    Y: ndarray = None,
    similarities_out: ndarray = None,
    chunk_size: int = 100,
) -> Similarity:
    """Initializes a concrete subclass of Similarity.

    Parameters
    ----------
    similarity:
        If pandas.DataFrame, see
        pymmunomics.sim.similarity.SimilarityFromFunction. If
        numpy.ndarray, see pymmunomics.sim.similarity.SimilarityFromFunction.
        If str, see pymmunomics.sim.similarity.SimilarityFromFunction.
        If Callable, see pymmunomics.sim.similarity.SimilarityFromFunction.
    X, Y, similarities_out:
        Only relevant for pymmunomics.sim.similarity.SimilarityFromFunction.
    chunk_size:
        See pymmunomics.sim.similarity.SimilarityFromFunction,
        or pymmunomics.sim.similarity.SimilarityFromFile. Only
        relevant if a callable or str is passed as `similarity`.

    Returns
    -------
    An instance of a concrete subclass of Similarity.
    """
    if isinstance(similarity, DataFrame):
        return SimilarityFromDataFrame(similarity=similarity)
    elif isinstance(similarity, ndarray):
        return SimilarityFromArray(similarity=similarity)
    elif isinstance(similarity, str):
        return SimilarityFromFile(
            similarity=similarity,
            chunk_size=chunk_size,
        )
    elif isinstance(similarity, Callable):
        return SimilarityFromFunction(
            similarity=similarity, X=X, Y=Y, similarities_out=similarities_out, chunk_size=chunk_size,
        )
    else:
        raise NotImplementedError(
            (
                "Type %s is not supported for argument "
                "'similarity'. Valid types include pandas.DataFrame, "
                "numpy.ndarray, numpy.memmap, str, or typing.Callable"
            )
            % type(similarity)
        )
