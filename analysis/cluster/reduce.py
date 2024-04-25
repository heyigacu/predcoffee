import os
import pandas as pd
import numpy as np
from rdkit import DataStructs,Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from sklearn import decomposition


base_dir = os.path.abspath(os.path.dirname(__file__))
# TODO
"""
please change smiles_path only
"""
filename = 'input.csv'
coordinates_save_path = 'tsne_coordinate.txt'
file_path = os.path.join(base_dir, filename)

coordinates_save_path = os.path.join(base_dir, coordinates_save_path)

def fp2arr(fp):
    """
    fingerprint -> array
    """
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

#fpMtx = np.array([fp2arr(fp) for fp in fps])

def tsne_reduce_dimension(fps):
    """
    tsne reduce dimension
    """
    tsne = TSNE(n_components=2)
    res = tsne.fit_transform(fps)
    np.savetxt(coordinates_save_path,res)
    return res

def pca_reduce_dimension(fps):
    pca = decomposition.PCA(n_components=2)
    pca_coords = pca.fit_transform(fps)
    np.savetxt(coordinates_save_path,pca_coords)
    return pca_coords


smileses = list(pd.read_csv(file_path, header=0, sep='\t')['Smiles'])
mols = [Chem.MolFromSmiles(smiles) for smiles in smileses]
fps = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)) for mol in mols])
coords = tsne_reduce_dimension(fps)




