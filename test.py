from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
IPythonConsole.ipython_useSVG = True
# 创建一个分子对象
smiles = 'CCO'  # 分子的SMILES表示
mol = Chem.MolFromSmiles(smiles)
# 生成Morgan指纹
radius = 2  # Morgan指纹的半径
bitvect = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius))
print(bitvect)
print(np.nonzero(bitvect))
onbits = np.nonzero(bitvect) 
for idx in onbits:
    atom_idx = onbits.index(idx)  # 第idx位在onbits中的索引
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    submol = Chem.PathToSubmol(mol, env)
    img = Draw.MolToImage(submol)
    img.show()
