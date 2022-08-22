'''
This code filter fasta, valid acid amin, length < 1020 (input of esm1b)
'''

import os
from Bio import SeqIO

class FastaFilter:

    def __init__(self, fa, prefix, is_train, is_rbp):

        if not os.path.isdir('fasta-filtered'):
            os.makedirs('fasta-filtered')

        if is_train:
            s1 = 'train'
        else:
            s1 = 'test'

        if is_rbp:
            s2 = 'RBP'
        else:
            s2 = 'nRBP'

        self.output_file = 'fasta-filtered/' + prefix + '.'+ s1 + '.'+ s2 + '.filtered.fa'

        self.fa = fa



    def filter_fasta(self):
        records = SeqIO.parse(self.fa, 'fasta')
        filtered1 = (rec for rec in records if all(
            ch in {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                   'Y'} for ch in rec.seq))
        filtered2 = (rec for rec in filtered1 if len(str(rec.seq)) < 1020)
        SeqIO.write(filtered2, self.output_file, 'fasta')
        return self.output_file
