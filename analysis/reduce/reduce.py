import os
import pandas as pd
import numpy as np
from rdkit import DataStructs,Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from sklearn import decomposition
import matplotlib.pyplot as plt

base_dir = os.path.abspath(os.path.dirname(__file__))
# TODO
"""
please change smiles_path only
"""
filename = 'cleaned_data.csv'
coordinates_save_path = 'pca_coordinate.txt'
coordinates_save_path = os.path.join(base_dir, coordinates_save_path)

file_path = os.path.join(base_dir, filename)


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

df = pd.read_csv(file_path, header=0, sep='\t')
smileses = list(df['Smiles'])
mols = [Chem.MolFromSmiles(smiles) for smiles in smileses]
fps = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)) for mol in mols])
coords = pca_reduce_dimension(fps)

coords = np.loadtxt(coordinates_save_path)

color_ls = []
for index,row in df.iterrows():
    if row['Label'] == 0:
        color_ls.append("#4DBBD6")
    else:
        color_ls.append("#E64B35")
df.insert(len(df.columns), 'Color', color_ls)
df.insert(len(df.columns), 'tSNE1', coords[:, 0])
df.insert(len(df.columns), 'tSNE2', coords[:, -1])
print(df)
df.plot.scatter('tSNE1', 'tSNE2', c='Color')
plt.savefig("/home/hy/Documents/Project/odor/coffee/analysis/reduce/cleaned_data.png", dpi=600)
plt.show()






