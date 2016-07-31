import numpy as np
import statistics
import pandas as pd


class KNearestNeighbors:
    @staticmethod
    def knn(xTr, yTr, xTe, k):
        """
        :param xTr: n x d matrix of known digit image vectors
        :param yTr: 1 x n matrix of known values
        :param xTe: m x d matrix of unknown digit image vectors
        :param k: number of nearest neighbors to use
        :return: predictions for each digit vector
        """
        # I is indices
        (I, D) = KNearestNeighbors.find_knn(xTr, xTe, k)
        (m, k) = I.shape
        preds = np.zeros(m)
        for r in range(m):
            l = []
            for c in range(k):
                l.append(yTr[I[r, c]])
            preds[r] = statistics.mode(l)

        return preds

    @staticmethod
    def find_knn(xTr, xTe, k):
        """
        :param xTr: n x d matrix of known digit image vectors
        :param xTe: m x d matrix of unknown digit image vectors
        :param k: number of nearest neighbors to find
        :return: (I, D): I is an m x k where I[i, j] is the jth closest vector in xTr to
        vector i in xTe. D has the corresponding distances  """
        all_distances = KNearestNeighbors.l2_distance(xTe, xTr)
        # Find the indices of the lowest distance.
        # Also, exclude the zero index distance because that is the distance of each vector to itself
        I = np.argsort(all_distances)[:, 1:(k+1)]
        D = np.sort(all_distances)[:, 1:(k+1)]
        return I, D

    @staticmethod
    def l2_distance(m1, m2):
        """
        Returns n x m l2_distance matrix where [i,j] is the distance between the ith vector in m1 and the jth vector in m2.
        :param m1: n x d numpy matrix
        :param m2: m x d numpy matrix
        :return: n x m Euclidean Product Matrix
        """
        ipm1 = np.sum(np.square(m1), axis=1)
        ipm2 = np.sum(np.square(m2), axis=1).transpose()
        gram = KNearestNeighbors.inner_product_matrix(m1, m2)
        return ipm1+ipm2 - (2*gram)

    @staticmethod
    def inner_product_matrix(m1, m2):
        """
        :param m1: n x d numpy matrix
        :param m2: m x d numpy matrix
        :return: n x m Gram matrix
        """
        return m1 * m2.transpose()

if __name__ == '__main__':
    df_train = pd.read_csv("kNN/res/DigitsTrain")
    df_test = pd.read_csv("kNN/res/DigitsTest")
    xTe = df_test.as_matrix()
    xTr = df_test.as_matrix()
