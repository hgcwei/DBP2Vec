import torch
import numpy as np
from Bio import SeqIO
import os
import pathTool

class ESModel:

    def __init__(self, fa):

        self.fa = fa
        self.folder = os.path.splitext(fa.split('/')[-1])[0]

    # def get_esm_pts_for_fasta(self):
    #     cmd = 'python extract.py esm1b_t33_650M_UR50S ' + self.fa + ' ' + self.dir_save + ' --include mean'
    #
    #     os.system(cmd)

    # def get_esm_fea_from_pt(self, ptf):
    #     x = torch.load(self.folder + ptf)
    #     val = x['mean_representations'][33]
    #     return val.numpy()


    # def get_esm_feas_from_pts(self):
    #     fls = os.listdir(self.dir_save)
    #     num = len(fls)
    #     feas = []
    #     labels = []
    #     for i in range(num):
    #         fea = self.get_esm_fea_from_pt(fls[i])
    #         feas.append(fea)
    #         labels.append(self.label)
    #
    #     return np.array(feas), labels

    def esm2feas(self):
        feas = []
        pt = pathTool.pathTool()
        for rec in SeqIO.parse(self.fa,'fasta'):
            ptf = pt.valid_filename(rec.description)
            if len(ptf)>165:
                ptf = ptf[:165]
            x = torch.load(self.folder + '/' + ptf + '.pt')
            val = x['mean_representations'][33]
            fea = val.numpy()
            feas.append(fea.tolist())
        return np.array(feas)

