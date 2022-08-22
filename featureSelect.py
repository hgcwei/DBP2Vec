from sklearn.feature_selection import SelectKBest,mutual_info_classif
from mrmr import mrmr_classif
import pandas as pd



class FeatureSelect:

    def __init__(self,train_data,train_label, dim):

        self.train_data = train_data
        self.train_label = train_label
        self.dim = dim

    def build_filter(self):

        skb = SelectKBest(mutual_info_classif,k=self.dim)

        skb = skb.fit(self.train_data,self.train_label)

        return skb.get_support(indices=True)

        # data_new = skb.transform(self.data)

        # return data_new

    def build_filter2(self):

       X = pd.DataFrame(self.train_data)
       y = pd.Series(self.train_label)

       selected_features = mrmr_classif(X,y,K = self.dim)
       print(selected_features)

       return selected_features




