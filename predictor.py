import argparse
import torch
import os
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pandas as pd
from model import MLP
from feature import morgan_featurizer
import joblib
from model import MPNN,MLP
from feature import Graph_smiles
import numpy as np
import dgl
work_dir = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='molecules odor predictor')
parser.add_argument("-a", "--algorithm", type=str, choices=['mlp','svm','rf','kpgt','mpnn'], default='mlp', help="predict algorithm")
parser.add_argument("-i", "--input", type=str, default=work_dir+'/input.csv', help="input file")
parser.add_argument("-o", "--output", type=str, default=work_dir+'/result.csv',help="output file")
args = parser.parse_args()

##################
# predict
##################

smileses = list(pd.read_csv(args.input, header=0, sep='\t')['Smiles'])
algorithm = args.algorithm

def collate_smiles(graphs):
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph

total = []
for smiles in smileses:
    if algorithm == 'mlp':
        try:
            n_feats = 2048
            n_hiddens = 256
            n_tasks = 2
            model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
            state_dict = torch.load(os.path.join(work_dir,'pretrained/all_mlp.pth'), map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            for i in list(DataLoader([morgan_featurizer(smiles)], batch_size=1, shuffle=False, collate_fn=None, drop_last=False)):
                features = i
            rst = model(features)
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['non-coffee','coffee']
            total.append([labels[rst.argmax()],rst[0],rst[1]])
        except:
            total.append(['error-smiles','nan','nan'])
    elif algorithm == 'kpgt':
        try:
            from kpgt.main import predict
        except:
            raise ValueError('please go to kpgt/main.py to predict smiles')
        try:
            rst = predict(smiless=[smiles])
            rst = rst.mean()
            labels = ['non-coffee','coffee']
            if rst > 0.5:
                total.append(['coffee',0, 1])
            else:
                total.append(['non-coffee',1,0])
        except:
            total.append(['error-smiles','nan','nan'])
    elif algorithm == 'svm' :
        try:
            model = joblib.load(os.path.join(work_dir,'pretrained/all_svm.pth'))
            rst = model.predict_proba([morgan_featurizer(smiles)])[0]
            labels = ['non-coffee','coffee']
            total.append([labels[np.argmax(rst)],rst[0],rst[1]])
        except:
            total.append(['error-smiles','nan','nan'])
    elif algorithm == 'rf':
        try:
            model = joblib.load(os.path.join(work_dir,'pretrained/all_rf.pth'))
            rst = model.predict_proba([morgan_featurizer(smiles)])[0]
            labels = ['non-coffee','coffee']
            total.append([labels[np.argmax(rst)],rst[0],rst[1]])
        except:
            total.append(['error-smiles','nan','nan'])
    elif algorithm == 'mpnn':
        try:
            n_feats = 74
            edge_in_feats = 12
            n_tasks = 2
            node_out_feats = 256
            edge_hidden_feats = 256
            batchsize = 32
            model = MPNN(node_in_feats=n_feats, edge_in_feats=edge_in_feats, node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats, n_tasks=n_tasks)
            state_dict = torch.load(os.path.join(work_dir,'pretrained/all_mpnn.pth'), map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            for i in list(DataLoader([Graph_smiles(smiles)], batch_size=1, shuffle=False, collate_fn=collate_smiles, drop_last=False)):
                graphs = i
            rst = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['non-coffee','coffee']
            total.append([labels[rst.argmax()],rst[0],rst[1]])
        except:
            total.append(['error-smiles','nan','nan'])


df = pd.DataFrame(total)
df.columns = ['Odor', 'non-coffee', 'coffee']
df.insert(0,'Smiles',smileses)
df.to_csv(args.output,index=False,header=True,sep='\t')
