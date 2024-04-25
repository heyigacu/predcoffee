import sys
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import random
from featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from dataset import FinetuneMoleculeDataset, Collator_tune
from model import LiGhTPredictor as LiGhT
from trainer import FinetuneTrainer
from utils import set_random_seed, Result_Tracker, Evaluator, PolynomialDecayLR
import warnings
import os
from load_data import scaffold_split
warnings.filterwarnings("ignore")


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

def predict(seed=32, 
             kfold=5,
             config_name='base', 
             model_dir='./pretrained/',
             model_name='dpp4',
             dataset='dpp4test',
             data_path='./datasets/',
             dataset_type='classification',
             metric='rocauc',
             dropout=0,
             n_threads=8,):
    set_random_seed(seed=32)
    ls = []
    save_path = model_dir + f'/{model_name}/{model_name}-0.pth'
    print(save_path)
    config = config_dict[config_name]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = Collator_tune(config['path_length'])
    test_dataset = FinetuneMoleculeDataset(root_path=data_path, dataset=dataset, dataset_type=dataset_type, split_name=f'scaffold-5', split='val')
    # print(test_dataset.mean, test_dataset.std)
    # test_dataset = FinetuneMoleculeDataset(root_path=data_path, dataset=dataset, dataset_type=dataset_type, split_name='scaffold-5', split=None)
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
    # evaluator = Evaluator(dataset, metric, test_dataset.n_tasks)
    # print(evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all)))

if __name__ == '__main__':
    predict(seed=32, 
            kfold=5,
            config_name='base', 
            model_dir='./pretrained/',
            dataset='dpp4',
            model_name='dpp4',
            data_path='./datasets/',
            dataset_type='classification',
            metric='rocauc',
            dropout=0,
            n_threads=8)
