from numpy import (
    array,
    allclose,
    dtype,
    memmap,
)
from pandas import DataFrame

# from pymmunomics.sim.similarity import calculate_similarity_matrix

# class TestCalculateSimilarityMatrix:
#     def test_defaults(self, tmp_path):
#         query = DataFrame({"cdr3_aa": ["CARDYW", "CARYFDYW", "CARDVW"]})
#         expected_similarity_matrix = array(
#             [
#                 [1.0, 0.09, 0.3],
#                 [0.09, 1.0, 0.027],
#                 [0.3, 0.027, 1.0],
#             ]
#         )
#         similarities_filepath = tmp_path / "similarity_matrix.npy"

#         calculate_similarity_matrix(
#             query=query,
#             similarities_filepath=similarities_filepath,
#         )
#         similarity_matrix = memmap(
#             similarities_filepath,
#             dtype=dtype("f8"),
#             mode="r+",
#             offset=0,
#             shape=(len(query), len(query)),
#             order="C",
#         )

#         assert allclose(similarity_matrix, expected_similarity_matrix)

#     def test_subject(self, tmp_path):
#         query = DataFrame({"cdr3_aa": ["CARDYW", "CARYFDYW", "CARDVW"]})
#         subject = DataFrame({"cdr3_aa": ["CARDYW", "CARDDDYW"]})
#         expected_similarity_matrix = array(
#             [
#                 [1.0, 0.09],
#                 [0.09, 0.09],
#                 [0.3, 0.027],
#             ]
#         )
#         similarities_filepath = tmp_path / "similarity_matrix.npy"

#         calculate_similarity_matrix(
#             query=query,
#             similarities_filepath=similarities_filepath,
#             subject=subject,
#         )
#         similarity_matrix = memmap(
#             similarities_filepath,
#             dtype=dtype("f8"),
#             mode="r+",
#             offset=0,
#             shape=(len(query), len(subject)),
#             order="C",
#         )

#         assert allclose(similarity_matrix, expected_similarity_matrix)


#     def test_seq_col(self, tmp_path):
#         query = DataFrame({
#             "cdr3_aa": ["CARDYW", "CARYFDYW", "CARDVW"],
#             "other_cdr3_aa": ["CARDY", "ARYFDYW", "ARDV"],
#         })
#         expected_similarity_matrix = array(
#             [
#                 [1.0, 0.0081, 0.09],
#                 [0.0081, 1.0, 0.0081],
#                 [0.09, 0.0081, 1.0],
#             ]
#         )
#         similarities_filepath = tmp_path / "similarity_matrix.npy"

#         calculate_similarity_matrix(
#             query=query,
#             similarities_filepath=similarities_filepath,
#             seq_col="other_cdr3_aa",
#         )
#         similarity_matrix = memmap(
#             similarities_filepath,
#             dtype=dtype("f8"),
#             mode="r+",
#             offset=0,
#             shape=(len(query), len(query)),
#             order="C",
#         )

#         assert allclose(similarity_matrix, expected_similarity_matrix)


#     def test_base(self, tmp_path):
#         query = DataFrame({"cdr3_aa": ["CARDYW", "CARYFDYW", "CARDVW"]})
#         expected_similarity_matrix = array(
#             [
#                 [1.0, 0.25, 0.5],
#                 [0.25, 1.0, 0.125],
#                 [0.5, 0.125, 1.0],
#             ]
#         )
#         similarities_filepath = tmp_path / "similarity_matrix.npy"

#         calculate_similarity_matrix(
#             query=query,
#             similarities_filepath=similarities_filepath,
#             base=0.5,
#         )
#         similarity_matrix = memmap(
#             similarities_filepath,
#             dtype=dtype("f8"),
#             mode="r+",
#             offset=0,
#             shape=(len(query), len(query)),
#             order="C",
#         )

#         assert allclose(similarity_matrix, expected_similarity_matrix)

#     def test_binding_similarity_kwargs(self, tmp_path):
#         query = DataFrame({
#             "cdr3_aa": ["CARDYW", "CARYFDYW", "CARDVW", "CARYDVW", "CARDDYYDYW"],
#             "match_col": ["match1", "match1", "match1", "match2", "match2"],
#             "set_intersection_col": [
#                 {"match", "mismatch1", "mismatch2"},
#                 {"match", "mismatch3"},
#                 {"mismatch4"},
#                 {"match", "mismatch5"},
#                 {"mismatch6", "mismatch7"},
#             ],
#         })
#         expected_similarity_matrix = array(
#             [
#                 [1.0, 0.09, 0.0, 0.0, 0.0],
#                 [0.09, 1.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 1.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 1.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 1.0],
#             ]
#         )
#         similarities_filepath = tmp_path / "similarity_matrix.npy"

#         calculate_similarity_matrix(
#             query=query,
#             similarities_filepath=similarities_filepath,
#             binding_similarity_kwargs={
#                 "match_cols": ["match_col"],
#                 "set_intersection_cols": ["set_intersection_col"],
#             }
#         )
#         similarity_matrix = memmap(
#             similarities_filepath,
#             dtype=dtype("f8"),
#             mode="r+",
#             offset=0,
#             shape=(len(query), len(query)),
#             order="C",
#         )

#         assert allclose(similarity_matrix, expected_similarity_matrix)

#     def test_binding_similarity_kwargs_override(self, tmp_path):
#         query = DataFrame({
#             "cdr3_aa": ["CARDYW", "CARYFDYW", "CARDVW"],
#             "alternative_cdr3_aa": ["ARDY", "CARYFDYW", "CARDVW"],
#         })
#         expected_similarity_matrix = array(
#             [
#                 [1.0, 0.09, 0.3],
#                 [0.09, 1.0, 0.027],
#                 [0.3, 0.027, 1.0],
#             ]
#         )
#         similarities_filepath = tmp_path / "similarity_matrix.npy"

#         calculate_similarity_matrix(
#             query=query,
#             similarities_filepath=similarities_filepath,
#             seq_col="alternative_cdr3_aa",
#             base=0.5,
#             binding_similarity_kwargs={
#                 "seq_col": "cdr3_aa",
#                 "base": 0.3,
#             }
#         )
#         similarity_matrix = memmap(
#             similarities_filepath,
#             dtype=dtype("f8"),
#             mode="r+",
#             offset=0,
#             shape=(len(query), len(query)),
#             order="C",
#         )

#         assert allclose(similarity_matrix, expected_similarity_matrix)