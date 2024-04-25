
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim

from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('TkAgg')
import copy
import joblib

from early_stop import EarlyStopping
from utils import *
from copy import deepcopy

PWD = os.path.abspath(os.path.dirname(__file__))


class train_mlp(object):
    ##############
    ## Binary-classify
    ##############    
    @staticmethod
    def train_bi_classify_kfolds(orimodel, kfolds=None, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='mlp_bi.pth', lr=0.001, weight_decay=0):
        val_metrics = []
        best_epochs = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = deepcopy(orimodel)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_features,train_labels) in enumerate(train_loader):
                    features, labels = train_features.to(device), train_labels.to(device)
                    preds = model(features)
                    loss = CrossEntropyLoss()(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_features, val_labels in val_loader:
                        features, labels = val_features.to(device), val_labels.to(device)
                        preds = model(features)
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = bi_classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(epoch, loss_val, model, metrics_val)
                if early_stopping.early_stop:
                    val_metrics.append(early_stopping.best_metrics)
                    best_epochs.append(early_stopping.best_epoch)
                    print("Early stopping")
                    break
        return np.array(val_metrics).mean(0), np.array(best_epochs).mean(0)
    @staticmethod
    def train_bi_classify_all(model, all=None, epochs=500, save_folder=PWD+'/pretrained/',save_name='mlp_bi.pth', lr=0.001, weight_decay=0):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(1, epochs+1):
            loss_train = 0.
            acc_train = 0.
            model.train()
            for batch_idx,(train_features,train_labels) in enumerate(all):
                features, labels = train_features.to(device), train_labels.to(device)
                logits = model(features)
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                acc = my_acc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                acc_train += acc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            torch.save(model.state_dict(), save_folder+save_name)
            if epoch%1 == 0:
                print('loss:',loss_train,'ACC:',acc_train)    
    @staticmethod
    def test_bi_classify(model, test=None, plot_cm=True, save_path=PWD+'/pretrained/gnn.pth', classnames=['Coffee','Non-Coffee']):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            features,labels = i
        preds = model(features)
        preds = preds.detach().cpu().numpy()
        rst = bi_classify_metrics(labels, preds, plot_cm=plot_cm, save_path=save_path, classnames=classnames)
        return rst

class train_gnn(object):
    ##############
    ## Binary-classify
    ##############    
    @staticmethod
    def train_bi_classify_kfolds(orimodel, kfolds=None, edge=True, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        val_metrics = []
        best_epochs = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = deepcopy(orimodel)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_graphs,train_labels) in enumerate(train_loader):
                    graphs, labels = train_graphs.to(device), train_labels.to(device)
                    if edge:
                        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                    else:
                        preds = model(graphs, graphs.ndata.pop('h'))
                    loss = CrossEntropyLoss()(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_graphs, val_labels in val_loader:
                        graphs, labels = val_graphs.to(device), val_labels.to(device)
                        if edge:
                            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                        else:
                            preds = model(graphs, graphs.ndata.pop('h'))
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = bi_classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(epoch, loss_val, model, metrics_val)
                if early_stopping.early_stop:
                    val_metrics.append(early_stopping.best_metrics)
                    best_epochs.append(early_stopping.best_epoch)
                    print("Early stopping")
                    break
        return np.array(val_metrics).mean(0), np.array(best_epochs).mean(0)
    @staticmethod
    def train_bi_classify_all(model, all=None, edge=True, epochs=500, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        rst = []
        for epoch in range(1, epochs+1):
            loss_train = 0.
            model.train()
            for batch_idx,(train_graphs,train_labels) in enumerate(all):
                graphs, labels = train_graphs.to(device), train_labels.to(device)
                if edge:
                    logits = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    logits = model(graphs, graphs.ndata.pop('h'))
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            torch.save(model.state_dict(), save_folder+save_name)
    @staticmethod
    def test_bi_classify(model, test=None, edge=True, save_path=PWD+'/pretrained/gnn.pth',classnames=['Bitter','Sweet']):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            graphs,labels = i
        if edge:
            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
        else:
            preds = model(graphs, graphs.ndata.pop('h'))
        preds = preds.detach().cpu().numpy()
        rst = bi_classify_metrics(labels, preds, plot_cm=True, save_path=save_path, classnames=classnames)
        return rst

class train_ml(object):
    @staticmethod
    def train_multi_classify_kfolds(orimodel, kfolds):
        metrics = []
        for (train_fold,val_fold) in kfolds:
            model = copy.deepcopy(orimodel)
            train_x,train_y = list(zip(*train_fold))
            train_x,train_y = np.array(list(train_x)),np.array(list(train_y))
            (val_x,val_y) = list(zip(*val_fold))
            val_x,val_y = np.array(list(val_x)),np.array(list(val_y))
            model.fit(train_x, train_y)
            preds = model.predict_proba(val_x)
            metrics.append(multi_classify_metrics(val_y,preds))
        return np.array(metrics).mean(0)
    def train_bi_classify_kfolds(orimodel, kfolds):
        metrics = []
        for (train_fold,val_fold) in kfolds:
            model = copy.deepcopy(orimodel)
            train_x,train_y = list(zip(*train_fold))
            train_x,train_y = np.array(list(train_x)),np.array(list(train_y))
            (val_x,val_y) = list(zip(*val_fold))
            val_x,val_y = np.array(list(val_x)),np.array(list(val_y))
            model.fit(train_x, train_y)
            preds = model.predict_proba(val_x)
            metrics.append(bi_classify_metrics(val_y,preds))
        return np.array(metrics).mean(0)
    def train_classify_all(model, all, save_path=PWD+'/pretrained/ml.pth'):
        train_x, train_y = list(zip(*all))
        train_x=list(train_x)
        train_y=list(train_y)
        model.fit(train_x, train_y)
        joblib.dump(model, save_path)
        
    def predict(features, save_path):
        model =  joblib.load(save_path)
        return model.predict_proba(features)
