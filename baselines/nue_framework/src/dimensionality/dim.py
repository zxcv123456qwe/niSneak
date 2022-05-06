from __future__ import print_function, division
import numpy as np
from scipy.spatial.distance import cdist


class Solver(object):
    def __init__(self, df):
        self.df = df
        self.num_samples = self.df.shape[0]
        self.samples = self.df.to_numpy()
        
    def compute_pairwise_l2distance(self):
        # print('ues L2 norm to compute the distance')
        R,C = np.triu_indices(self.num_samples,1)    # Denote N = num_samples
        pair_innerdot = np.einsum('ij,ij->i', self.samples[R,:], self.samples[C,:]) 
        # shape: (Nx(N-1)/2,) items are uptriangular part of mat [(Xi, Xj)], () denotes inner product
        norm = np.einsum('ij,ij->i', self.samples, self.samples)  # shape: (N,)
        return norm[R] + norm[C] - 2*pair_innerdot    
    
    def compute_pairwise_l1distance(self):
        # print('ues L1 norm to compute the distance')
        R, C = np.triu_indices(self.num_samples, 1)  # Denote N = num_samples
        # R, C contain the row indexs and column indexs of upper-triangular part
        # shape: (Nx(N-1)/2,)
        l1norm = np.abs(self.samples[R, :] - self.samples[C, :]).sum(-1)

        return l1norm
        
    def optimized_pairwise_l1distance(self):
        # print('ues L1 norm to compute the distance')
        """
        samples: matrix of size NxD
        Returns: NxN matrix D, with entry D_ij = manhattan or L1 distance between rows X_i and X_j
        """
        D = cdist(self.samples, self.samples, metric='cityblock')
        iu1 = np.triu_indices(self.num_samples)
        D = D.astype("float")
        D[iu1] = float('inf')
        # set the upper-triangular as Positive infinity
        return D
        
    def optimized_pairwise_l2distance(self):
        # print('ues L2 norm to compute the distance')
        """
        samples: matrix of size NxD
        Returns: NxN matrix D, with entry D_ij = squared euclidean distance between rows X_i and X_j
        """
        # Math? See https://stackoverflow.com/questions/37009647
        sum_X = np.sum(np.square(self.samples), 1)
        D = np.add(np.add(-2 * np.dot(self.samples, self.samples.T), sum_X).T, sum_X)
        # **0.5 ?
        iu1 = np.triu_indices(self.num_samples)
        D = D.astype("float")
        D[iu1] = float('inf')
        # set the upper-triangular as Positive infinity
        return D

    def optimized_compute_Cr(self, distances, r):
        return np.sum(distances < r) / (0.5*self.num_samples*(self.num_samples-1))

        
    def compute_Cr(self, distances, r):
        return np.sum(distances < r) / len(distances)
        
    def show_curve(self, logrs, version=1):
        start, end, step = logrs.split(":")
        assert int(step) > 0
        logrs = np.linspace(float(start), float(end), num=int(step))
        rs = np.exp(logrs)
        # distances = self.compute_pairwise_l1distance()
        
        # if version == 1:
        #     distances = self.compute_pairwise_l1distance()
        # else:
        #     distances = self.compute_pairwise_l2distance()
        
        if version == 1:
            distances = self.optimized_pairwise_l1distance()
        else:
            distances = self.optimized_pairwise_l2distance()

        logCrs = []
        for r in rs:
            logCrs.append(self.optimized_compute_Cr(distances, r))
            # logCrs.append(self.compute_Cr(distances, r))
        logCrs = np.log(np.array(logCrs))
        logCrs_d = (logCrs - logCrs[[*range(1,len(logCrs)), -1]]) / (logrs[0] - logrs[1])
        logCrs_d = logCrs_d[~np.isnan(logCrs_d)]
        logCrs_d = logCrs_d[np.isfinite(logCrs_d)]
        # remove the nan and inf from logCrs_d
        # print("candidate estiamted instrinsic dim: {}".format(logCrs_d))
        return np.max(logCrs_d)

