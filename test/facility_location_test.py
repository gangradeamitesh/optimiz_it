# import unittest
# from optimiz.facility_location import FacilityLocation

# class TestFacilityLocation(unittest.TestCase):

#     def test_compute_simi_martrix(self):
#         X = [[1,2] , [3,4]]
#         f1 = FacilityLocation(X , None)
#         expected_output = [[0,8] , [8,0]]
#         self.assertEqual()

from sklearn.metrics import pairwise_distances
X = [[1, 2], [3, 4], [5, 6]]
X_pairwise = pairwise_distances(X , metric="euclidean" , squared=True)
print(X_pairwise.shape[0])
print(X_pairwise.shape[1])
print(X_pairwise.ndim)
print(X_pairwise)