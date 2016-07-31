import unittest
import numpy as np
from kNN import KNearestNeighbors as knn


class TestKNN(unittest.TestCase):
    def test_l2_distance(self):
        a = np.matrix([[1, 2], [3, 4], [9, 10]])
        b = np.matrix([[5, 6], [7, 8]])
        c = np.matrix([[32, 72], [8, 32], [32, 8]])
        l2 = knn.l2_distance(a, b)
        print(l2)
        equal = np.array_equal(l2, c)
        self.assertEqual(True, equal)


if __name__ == '__main__':
    TestKNN.main()