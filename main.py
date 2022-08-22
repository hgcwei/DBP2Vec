import featureModel
import os
import numpy as np
import featureSelect
import featureUniform
import svModel
import fastaFilter

TNAME = 'swi'
svm_dir = 'svm/' + TNAME
c = 2
g = 0.5
kb = 150
# ---------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Data precessing,    Input:  four fasta in folder fasta,  two for train and two for test          Output:   four fasta



f1 = 'fasta/UniSwiss-Tr-posi.fasta'
f2 = 'fasta/UniSwiss-Tr-nega.fasta'

f3 = 'fasta/UniSwiss-Tst-posi.fasta'
f4 = 'fasta/UniSwiss-Tst-nega.fasta'

ff = fastaFilter.FastaFilter(f1,TNAME,True,True)
f1_ = ff.filter_fasta()

ff = fastaFilter.FastaFilter(f2,TNAME,True,False)
f2_ = ff.filter_fasta()

ff = fastaFilter.FastaFilter(f3,TNAME,False,True)
f3_ = ff.filter_fasta()

ff = fastaFilter.FastaFilter(f4,TNAME,False,False)
f4_ = ff.filter_fasta()

# -------------------------------------------------------------------------------------------------------------------------------------------------
#  Step 2: Generate features pt files by esm1b.   open command window


# cd RBP2Vec
# python extract.py esm1b_t33_650M_UR50S fasta-filtered/sal.test.nRBP.filtered.fa sal.test.nRBP.filtered/  --include mean
# NOTE: Please use our updated extract.py here, not its original version in https://github.com/facebookresearch/esm
#
#

# python extract.py esm1b_t33_650M_UR50S fasta-filtered/swi.train.nRBP.filtered.fa swi.train.nRBP.filtered/  --include mean  --toks_per_batch 2000


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# #Step 3: Generate features matrix from pt files

fm = featureModel.FeatureModel(f1_)
train_data_pos = fm.get_feas_fasta()
train_label_pos = fm.get_labels(1)

fm = featureModel.FeatureModel(f2_)
train_data_neg = fm.get_feas_fasta()
train_label_neg = fm.get_labels(0)

fm = featureModel.FeatureModel(f3_)
test_data_pos = fm.get_feas_fasta()
test_label_pos = fm.get_labels(1)

fm = featureModel.FeatureModel(f4_)
test_data_neg = fm.get_feas_fasta()
test_label_neg = fm.get_labels(0)


train_labels = train_label_pos + train_label_neg
test_labels = test_label_pos + test_label_neg
labels_all = train_labels + test_labels
train_num = len(train_labels)
label_svm = np.array(labels_all)

train_data = np.vstack((train_data_pos,train_data_neg))
test_data = np.vstack((test_data_pos,test_data_neg))

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4: Feature selection by mutual information

ff = featureSelect.FeatureSelect(train_data,train_labels,kb)
kbest_index = ff.build_filter()

train_data_svm = train_data[:,kbest_index]
test_data_svm = test_data[:,kbest_index]

train_svm_label = label_svm[:train_num]
test_svm_label = label_svm[train_num:]

# --------------------------------------------------------------------------------------------------------------------------------------------
# Step 5: Convert data to libsvm format

if not os.path.isdir('svm'):
    os.makedirs('svm')

if not os.path.isdir(svm_dir):
    os.makedirs(svm_dir)

fu = featureUniform.FeatureUniform()

fu.libsvm_form(train_data_svm,train_svm_label,svm_dir + '/train')

fu = featureUniform.FeatureUniform()

fu.libsvm_form(test_data_svm,test_svm_label,svm_dir + '/test')


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Step 6: SVM classification, including scale, train, test and evaluation

sm = svModel.SVModel(svm_dir,c=c,g=g, kb = kb)
sm.svm_scale()
sm.svm_train()
sm.svm_predict()
sm.svm_evaluate()






