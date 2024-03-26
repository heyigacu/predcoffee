import os
import pandas as pd
from data_clean import data_preprocess, statistics
from load_data import *
from feature import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from model import MLP, MPNN
from trainer import train_mlp, train_ml, train_gnn
import numpy as np

parent_dir = os.path.abspath(os.path.dirname(__file__))
print(parent_dir)


data_path = parent_dir+"/dataset/train/cleaned_data_sampled.csv"

df = pd.read_csv(data_path,sep='\t',header=0)
smileses = df['Smiles']
labels = df['Label'].astype(int)
print('read data:', Counter(labels))
graphs = [Graph_smiles(smiles) for smiles in smileses]
nBits = 2048
features = np.array([morgan_featurizer(smiles, nBits=nBits) for smiles in smileses])

def mlp_bi_class():
    tuple_ls = list(zip(features, labels))
    #########
    # mlp
    #########
    batchsize = 32
    drop_last = True
    n_feats = 2048
    n_hiddens = 2000
    n_tasks = 2

    model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
    kfolds = BaseDataLoader.load_data_kfold_torch_batchsize(tuple_ls, batchsize=batchsize, Stratify=True, drop_last=drop_last, sample_method=None)
    print('mlp start training!')
    rst_mlp, best_epoch = train_mlp.train_bi_classify_kfolds(model, kfolds=kfolds, max_epochs=500, patience=7, save_folder=parent_dir+'/pretrained/',save_name='kfolds.pth')
    print('optimization finished!', rst_mlp)
    print(best_epoch)
    # train_mlp.train_bi_classify_all(model, all=tuple_ls, max_epochs=500, patience=7, save_folder=parent_dir+'/pretrained/',save_name='mlp_all.pth')
    return rst_mlp

def SVMGridSearchCV():
    import sklearn.model_selection as ms
    from sklearn import svm
    params = [{'kernel':['linear'], 'C':[0.01, 0.1, 1, 10, 100,]},
            {'kernel':['poly'], 'C':[0.01, 0.1, 1, 10, 100,], 'degree':[2, 3]}, 
            {'kernel':['rbf'], 'C':[0.01, 0.1, 1,10,100,1000], 'gamma':[1, 0.1, 0.01, 0.001]}]
    model = ms.GridSearchCV(svm.SVC(probability=True), params, cv=5)
    model.fit(features, labels)
    with open(parent_dir+"/analysis/model_result/SVMGridSearch.csv", 'w') as f:
        for p, s in zip(model.cv_results_['params'],model.cv_results_['mean_test_score']):
            f.write(str(p)+str(s)+'\n')
    print(model.best_params_)
    return model.best_estimator_

def svm_bi_class(svm_model):
    tuple_ls = list(zip(features, labels))
    #########
    # svm
    #########
    batchsize = int(len(tuple_ls)/16)
    kfolds = BaseDataLoader.load_data_kfold_numpy(tuple_ls, Stratify=True)
    print('svm start training!')
    rst_svm = train_ml.train_bi_classify_kfolds(orimodel=svm_model, kfolds=kfolds)
    print('optimization finished!', rst_svm)
    return rst_svm

def RFGridSearch():
    import sklearn.model_selection as ms
    from sklearn.ensemble import RandomForestClassifier
    params = [{'n_estimators':[20, 100, 500, 1000, 1500], 
                'min_samples_split':[2, 5, 20, 50], 
                "min_samples_leaf": [1, 5, 20, 50],
                'max_features':["sqrt", "log2", None],
                'max_depth':[3, 6, 12]}]
    model = ms.GridSearchCV(RandomForestClassifier(), params, cv=5)
    model.fit(features, labels)
    with open(parent_dir+"/analysis/model_result/RFGridSearch.csv", 'w') as f:
        for p, s in zip(model.cv_results_['params'],model.cv_results_['mean_test_score']):
            f.write(str(p)+str(s)+'\n')
    print(model.best_params_)
    return model.best_estimator_


def rf_bi_class(rf_model):
    tuple_ls = list(zip(features, labels))
    #########
    # rf
    #########
    from sklearn.ensemble import RandomForestClassifier
    kfolds = BaseDataLoader.load_data_kfold_numpy(tuple_ls, Stratify=True)
    print('rf start training!')
    rst_rf = train_ml.train_bi_classify_kfolds(orimodel=rf_model, kfolds=kfolds)
    print('optimization finished!', rst_rf)
    return rst_rf


def gnn_bi_class():
    tuple_ls = list(zip(graphs, labels))
    #########
    # svm
    #########
    n_feats = 74
    edge_in_feats = 12
    n_tasks = 2
    node_out_feats = 256
    edge_hidden_feats = 256
    # batchsize = int(len(tuple_ls)/16)
    batchsize = 32

    model = MPNN(node_in_feats=n_feats, edge_in_feats=edge_in_feats, node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats, n_tasks=n_tasks)
    kfolds = BaseDataLoader.load_data_kfold_graph_batchsize(tuple_ls, batchsize, Stratify=True, drop_last=True)
    print('mpnn start training!')
    rst_gnn, best_epoch = train_gnn.train_bi_classify_kfolds(model, kfolds=kfolds, edge=True, max_epochs=500, patience=7, save_folder=parent_dir+'/pretrained/',save_name='gnn.pth')
    print('optimization finished!', rst_gnn)
    return rst_gnn


def total():
    # SVMGridSearchCV()
    # RFGridSearch()
    total_rst = []
    for i in range(10):
        rst_mlp = mlp_bi_class()
        from sklearn import svm
        svm_paras = {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'probability':True}
        model =svm.SVC(**svm_paras)
        rst_svm = svm_bi_class(model)
        rf_paras = {'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 50, 'min_samples_split': 2, 'n_estimators': 100}
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**rf_paras)
        rst_rf = rf_bi_class(model)
        rst_gnn = gnn_bi_class()
        total_rst += [rst_mlp, rst_rf, rst_svm, rst_gnn]
    np.savetxt(parent_dir+"/analysis/model_result/performance.txt", np.array(total_rst))

total()
tuple_ls = list(zip(features, labels))
batchsize = int(len(tuple_ls)/16)
drop_last = True
n_feats = 2048
n_hiddens = 256
n_tasks = 2
model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
all = BaseDataLoader.load_data_all_torch_batchsize(tuple_ls, batchsize, drop_last=True)
rst_mlp = train_mlp.train_bi_classify_all(model, all=all, epochs=3, save_folder=parent_dir+'/pretrained/',save_name='all_mlp.pth')
