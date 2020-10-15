# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:38:02 2020

@author: Sai Anil Kumar M
"""

import pandas as pd
import sklearn.model_selection as model_selection

if __name__ == "__main__":
    # train data is in train.csv
    df = pd.read_csv('train.csv')
    
    # create a column called kfold and fill it with -1
    df['kfold'] = -1
    
    # the next step is to randomize the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # assuming the target to be predicted is in column called Target
    # fetch targets
    y = df.target.values
    
    # initiate kfold from model selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column with fold numbers
    for fold, (trn_, val_) in enumerate(kf.split(X=df,y=y)):
        df.loc[val_, 'kfold'] = fold
    
    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)