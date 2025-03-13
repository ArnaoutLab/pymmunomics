"""Module for calculating (weighted) similarities between community sequences and known other sequences.

Similarities can be provided in the form of a numpy array, a pandas DataFrame, or a csv-formatted file;
or they can be calculated on the fly using a client-supplied function.

Uses the `greylock` package. See `greylock` documentation for further info.

Functions
---------
binding_similarity:
    Calculates binding similarities of antibodies or B-/T-cell
    receptors.
make_similarities:
    Instantiates the appropriate greylock Similarity class for the
    given arguments.
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

from greylock.similarity import Similarity, SimilarityFromDataFrame, SimilarityFromArray, IntersetSimilarityFromFile
from greylock.ray import IntersetSimilarityFromRayFunction


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
        return 0.3 ** levenshtein(left[0], right[0])

class SimilarityWrapper:
    """
    Wrapper class to adapt `greylock` objects for the weighted_similarity functionality.
    """
    def __init__(self, similarity: Similarity):
        self.similarity = similarity

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
        return self.similarity.weighted_abundances(species_frequencies)

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
            A similarity matrix, a path to a similarity matrix, or a
            function that can be called on pairs of species to calculate
            similarities. Similarity matrix rows correspond to query
            species and columns to subject species.
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
        return SimilarityWrapper(SimilarityFromDataFrame(similarity))
    elif isinstance(similarity, ndarray):
        return SimilarityWrapper(SimilarityFromArray(similarity=similarity))
    elif isinstance(similarity, str):
        return SimilarityWrapper(IntersetSimilarityFromFile(
            similarity_file_path=similarity,
            chunk_size=chunk_size,
        ))
    elif isinstance(similarity, Callable):
        return SimilarityWrapper(IntersetSimilarityFromRayFunction(
            func=similarity,
            X=X,
            Y=Y,
            similarities_out=similarities_out,
            chunk_size=chunk_size,
        ))
    else:
        raise NotImplementedError(
            (
                "Type %s is not supported for argument "
                "'similarity'. Valid types include pandas.DataFrame, "
                "numpy.ndarray, numpy.memmap, str, or typing.Callable"
            )
            % type(similarity)
        )
