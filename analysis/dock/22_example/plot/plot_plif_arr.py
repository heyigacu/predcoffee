
# select 7A, byres ligand around 7
# None
# Hydrophobic
# Aromatic Face to Face
# Aromatic Edge to Face
# H-bond (protein is donor)
# H-bond (protein is acceptor)
# Electrostatic (protein +)
# Electrostatic (protein -)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
base_dir = os.path.abspath(os.path.dirname(__file__)) 


one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    
'GLY':'G', 'PRO':'P', 'CYS':'C'}
def three2one_aa(three_aa):
    return one_letter[three_aa[0:3]]+three_aa[3:]
color_list = [ 'aliceblue','#C82423','darkorange','gold','green','cyan','blue','purple']

plif_path = base_dir+'/../check_plif_hippos.csv'
aas = []
with open(os.path.join(base_dir, 'aa.txt'),'r') as f:
    lines = f.readlines()
    for line in lines:
        aas.append(three2one_aa(line.strip()))            
df = pd.read_csv(plif_path,sep='\t',header=0)
# idxs = range(df_nk.shape[0]-1,df.shape[0],1)
idxs = list(df['Idx'])
print(idxs)
# idxs = df[df['Group'] == 19]['No'] 
arr = np.loadtxt(os.path.join(base_dir, 'processed_plif.txt')).transpose()
plt.figure(figsize=(8,5))
print(arr.shape)
cmap = sns.color_palette(color_list)
xindice = np.arange(0.5,float(len([idxs]))+0.5,1)
yindice = np.arange(0.5,float(len(aas))+0.5,1)
ax = sns.heatmap( arr, cmap=cmap,) # cbar=False
plt.yticks(yindice, aas, rotation=0, size=8) 
# plt.xticks(xindice, rotation=0, size=8) 
plt.xticks([]) 
plt.ylabel('residue')
cbar = ax.collections[0].colorbar
# cbar.set_ticks(np.arange(0.5,7.5,7/8))  
cbar.set_ticklabels(['None', 'Hydrophobic', 'Aromatic Face to Face', 'Aromatic Edge to Face', 'H-bond (protein is donor)','H-bond (protein is acceptor)',
                     'Electrostatic (protein +)', 'Electrostatic (protein -)'])  
plt.tight_layout()
plt.savefig(base_dir+'/'+'a.png')
plt.show()
