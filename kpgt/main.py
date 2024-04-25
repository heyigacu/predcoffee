import sys
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import pandas as pd
import random
from featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES, smiles_to_graph_tune
from dataset import FinetuneMoleculeDataset, preprocess_batch_light, Collator_tune
from model import LiGhTPredictor as LiGhT
from trainer import FinetuneTrainer
from utils import set_random_seed, Result_Tracker, Evaluator, PolynomialDecayLR
import warnings
import os
from load_data import scaffold_split
warnings.filterwarnings("ignore")
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
import dgl.backend as F
import dgl
from rdkit import Chem
from scipy import sparse as sp
from descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from multiprocessing import Pool
from torch.utils.data import Dataset

config_dict = {
    'base': {
        'd_node_feats': 137, 'd_edge_feats': 14, 'd_g_feats': 768, 'd_hpath_ratio': 12, 'n_mol_layers': 12, 'path_length': 5, 'n_heads': 12, 'n_ffn_dense_layers': 2,'input_drop':0.0, 'attn_drop': 0.1, 'feat_drop': 0.1, 'batch_size': 1024, 'lr': 2e-04, 'weight_decay': 1e-6,
        'candi_rate':0.5, 'fp_disturb_rate': 0.5, 'md_disturb_rate': 0.5
    }
}

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers-2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)

def finetune(seed=32, 
             kfold=5,
             n_epochs=50, 
             config_name='base', 
             model_path='./pretrained/base.pth',
             dataset='predcoffee',
             data_path='./datasets/',
             dataset_type='classification',
             metric='rocauc',
             weight_decay=0.,
             dropout=0,
             lr=3e-5,
             n_threads=8,
             ):
    set_random_seed(seed=32)
    for i in range(kfold):
        model_save_dir = os.path.dirname(model_path)+f'/{dataset}/'
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        save_path = model_save_dir + f'{dataset}-{i}.pth'
        config = config_dict[config_name]
        vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        g = torch.Generator()
        g.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        collator = Collator_tune(config['path_length'])
        train_dataset = FinetuneMoleculeDataset(root_path=data_path, dataset=dataset, dataset_type=dataset_type, split_name=f'scaffold-{i}', split='train')
        val_dataset = FinetuneMoleculeDataset(root_path=data_path, dataset=dataset, dataset_type=dataset_type, split_name=f'scaffold-{i}', split='val')
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
        # Model Initialization
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=train_dataset.d_fps,
            d_md_feats=train_dataset.d_mds,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=dropout,
            feat_drop=dropout,
            n_node_types=vocab.vocab_size
        ).to(device)
        # Finetuning Setting
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(f'{model_path}').items()})
        del model.md_predictor
        del model.fp_predictor
        del model.node_predictor
        model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=train_dataset.n_tasks, n_layers=2, predictor_drop=dropout, device=device, d_hidden_feats=256)
        print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1e6))
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=n_epochs*len(train_dataset)//32//10, tot_updates=n_epochs*len(train_dataset)//32,lr=lr, end_lr=1e-9,power=1)
        if dataset_type == 'classification':
            loss_fn = BCEWithLogitsLoss(reduction='none')
        else:
            loss_fn = MSELoss(reduction='none')
        if dataset_type == 'classification':
            evaluator = Evaluator(dataset, metric, train_dataset.n_tasks)
        else:
            evaluator = Evaluator(dataset, metric, train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
        result_tracker = Result_Tracker(metric)
        summary_writer = None
        trainer = FinetuneTrainer(optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device,model_name='LiGhT', label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None, label_std=train_dataset.std.to(device) if train_dataset.std is not None else None)
        best_train, best_val = trainer.fit(model, train_loader, val_loader, save_path, n_epochs)
        print(f"train: {best_train:.3f}, val: {best_val:.3f}")

def evaluate(seed=32, 
             kfold=5,
             config_name='base', 
             model_dir='./pretrained/',
             model_name='dpp4',
             dataset='predcoffee',
             data_path='./datasets/',
             dataset_type='classification',
             metric='rocauc',
             dropout=0,
             n_threads=8,):
    from my_metrics import bi_classify_metrics,onehot
    set_random_seed(seed=32)
    ls = []
    for i in range(kfold):
        save_path = model_dir + f'/{model_name}/{model_name}-{i}.pth'
        print(save_path)
        config = config_dict[config_name]
        vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        g = torch.Generator()
        g.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        collator = Collator_tune(config['path_length'])
        test_dataset = FinetuneMoleculeDataset(root_path=data_path, dataset=dataset, dataset_type=dataset_type, split_name=f'scaffold-{i}', split='val')
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
        # Model Initialization
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=test_dataset.d_fps,
            d_md_feats=test_dataset.d_mds,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=dropout,
            feat_drop=dropout,
            n_node_types=vocab.vocab_size
        ).to(device)
        # Finetuning Setting
        model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=test_dataset.n_tasks, n_layers=2, predictor_drop=dropout, device=device, d_hidden_feats=256)
        del model.md_predictor
        del model.fp_predictor
        del model.node_predictor
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_path).items()})
        model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in test_loader:
            (smiles, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(device)
            md = md.to(device)
            g = g.to(device)
            labels = labels.to(device)
            predictions = model.forward_tune(g, ecfp, md)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        temp_labled = []
        for batch in labels_all:
            for bt in batch:
                temp_labled.append(list(bt)[0])
        temp_preded = []
        for batch in predictions_all:
            for bt in batch:
                temp_preded.append((list(bt)[0]))
                if (list(bt)[0])>0.5 :
                    temp_preded.append(1)
                else :
                    temp_preded.append(0)
        preds = onehot(np.array(temp_preded),2)
        labels = onehot(np.array(temp_labled),2)
        ls.append(bi_classify_metrics(labels, preds))
    print(np.mean(np.array(ls),axis=0))



class _FinetuneMoleculeDataset(Dataset):
    def __init__(self, smiless, graphs, fps, mds, path_length=5, n_virtual_nodes=2):
        # Dataset Setting
        self.n_tasks = 1
        self.mean = None
        self.std = None
        self.fps = fps
        self.mds = mds
        # self._task_pos_weights = self.task_pos_weights()
        self.smiless = smiless
        self.graphs = graphs
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
    def __len__(self):
        return len(self.smiless)
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx]
    # def task_pos_weights(self):
    #     task_pos_weights = torch.ones(self.labels.shape[1])
    #     num_pos = torch.sum(torch.nan_to_num(self.labels,nan=0), axis=0)
    #     masks = F.zerocopy_from_numpy((~np.isnan(self.labels.numpy())).astype(np.float32))
    #     num_indices = torch.sum(masks, axis=0)
    #     task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
    #     return task_pos_weights



class _Collator_tune(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
    def __call__(self, samples):
        smiles_list, graphs, fps, mds = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        return smiles_list, batched_graph, fps, mds



def predict(seed=32, 
             kfold=5,
             config_name='base', 
             model_dir='./pretrained/',
             model_name='predcoffee',
             dataset_type='classification',
             smiless=[],
             dropout=0,
             n_threads=8,):
    set_random_seed(seed=32)
    ls = []

    config = config_dict[config_name]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device("cpu")
    collator = _Collator_tune(config['path_length'])

    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune, smiless, max_length=5, n_virtual_nodes=2, n_jobs=32)  
    valid_ids = []
    valid_graphs = []
    for i, g_ in enumerate(graphs):
        if g_ is not None:
            valid_ids.append(i)
            valid_graphs.append(g_)

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    fps = torch.from_numpy(FP_sp_mat.todense().astype(np.float32))

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(5).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    md = arr[:,1:].astype(np.float32)
    mds = torch.from_numpy(np.where(np.isnan(md), 0, md))

    predict_dataset = _FinetuneMoleculeDataset(smiless, graphs, fps, mds)
    predict_loader = DataLoader(predict_dataset, batch_size=100, shuffle=False, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    # # Model Initialization
    for i in range(5):
        save_path = model_dir + f'/{model_name}-{i}.pth'
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=predict_dataset.d_fps,
            d_md_feats=predict_dataset.d_mds,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=dropout,
            feat_drop=dropout,
            n_node_types=vocab.vocab_size
        ).to(device)
        # Finetuning Setting
        model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=predict_dataset.n_tasks, n_layers=2, predictor_drop=dropout, device=device, d_hidden_feats=256)
        del model.md_predictor
        del model.fp_predictor
        del model.node_predictor
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_path, map_location=device).items()})
        model.eval()
        for batched_data in predict_loader:
            (smiles, g, ecfp, md) = batched_data
            ecfp = ecfp.to(device)
            md = md.to(device)
            g = g.to(device)
            predictions = model.forward_tune(g, ecfp, md)
            ls.append(predictions.detach().cpu()[0][0].numpy())
    return np.array(ls)
    

if __name__ == '__main__':
    finetune()
    evaluate()
    # predict(smiless=['CC(=C)C(CCC(=O)C)CC(=O)O','CCC1(C(=O)OCC)C(C)(c2ccccc2)O1'])
