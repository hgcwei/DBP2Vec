import numpy as np
import esModel


class FeatureModel:

    def __init__(self, fa):

        self.fa = fa
        self.feas = []

    def get_feas_fasta(self):
        em = esModel.ESModel(self.fa)
        self.feas = em.esm2feas()
        return self.feas

    def get_labels(self,label):
         labels = []
         m, n = self.feas.shape
         for i in range(m):
             labels.append(label)
         return labels

