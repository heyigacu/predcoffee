
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rdkit
from rdkit.Chem import Draw
from rdkit.Chem import AllChem 
from rdkit import Chem
import shap
from torch.utils.data.dataloader import DataLoader
from feature import morgan_featurizer
"""
https://shap-lrjball.readthedocs.io/en/latest/examples.html#gradient-explainer-examples
"""
parent_dir = os.path.abspath(os.path.dirname(__file__))
class MLP(nn.Module):
    def __init__(self, n_feats, n_hiddens, n_tasks):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feats, n_hiddens)
        self.activate1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.activate2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(n_hiddens, n_tasks)
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.dropout1(self.activate1(self.fc1(x)))        
        x = self.activate2(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x

def load_pretrained_model(model=MLP(n_feats = 2048, n_hiddens = 256, n_tasks = 2), path = parent_dir+"/pretrained/all_mlp.pth"):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    return model

def load_train_data():
    data_path = parent_dir+"/dataset/train/cleaned_data_sampled.csv"
    df = pd.read_csv(data_path,sep='\t',header=0)
    smileses = df[df['Odor']=='coffee']['Smiles']
    nBits = 2048
    train_features = np.array([morgan_featurizer(smiles, nBits=nBits) for smiles in smileses])
    train_features = DataLoader(train_features, batch_size=len(train_features), shuffle=True, collate_fn=None, drop_last=False)
    for i in train_features:
        train_features = i
    return train_features

def load_single_data_torch(smiles='CCCO'):
    for i in list(DataLoader([morgan_featurizer(smiles)], batch_size=1, shuffle=False, collate_fn=None, drop_last=False)):
        feature = i
    return feature

def generate_morgan_fp_bit(smiles, nBits=2048):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=nBits))

def generate_morgan_fp(smiles):
    info = {}
    fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles),  2, bitInfo=info)
    return info

def draw_bit(smiles, index):
    # bitvect = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)
    info = generate_morgan_fp(smiles)
    return Draw.DrawMorganBit(Chem.MolFromSmiles(smiles), index, info)



