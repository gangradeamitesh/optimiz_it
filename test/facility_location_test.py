import unittest
from optimiz.facility_location import FacilityLocation

class TestFacilityLocation(unittest.TestCase):

    def test_compute_simi_martrix(self):
        X = [[1,2] , [3,4]]
        f1 = FacilityLocation(X , None)
        expected_output = [[0,8] , [8,0]]
        self.assertEqual()