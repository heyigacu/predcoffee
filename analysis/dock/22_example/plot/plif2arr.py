
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.abspath(os.path.dirname(__file__)) 
plif_path = base_dir+'/../check_plif_hippos.csv'



one_letter = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    
'GLY':'G', 'PRO':'P', 'CYS':'C'}

def three2one_aa(three_aa):
    return one_letter[three_aa[0:3]]+three_aa[3:]

aas = []
with open(os.path.join(base_dir, 'aa.txt'),'r') as f:
    line = f.readlines()[0]
    words = line.split(' ')
    for word in words:
        aas.append(three2one_aa(word.strip()))

def PLIF(idx):
    df = pd.read_csv(plif_path,header=0,sep='\t')
    bitstring = df[df['Idx'] == idx]['PlifHippos'].values[0]
    bit_ls = list(str(bitstring))
    arr = []
    for i in bit_ls:
        arr.append(int(i))
    matrix = np.array(arr).reshape(-1,7)
    rst = []
    for row in matrix:
        one_index = list(np.where(row==1))[0]
        if len(one_index) == 0:
            rst.append(0)
        else:
            rst.append(max(one_index)+1)
    return np.array(rst).astype(int)

def PLIF_all():
    df = pd.read_csv(plif_path,header=0,sep='\t')
    idxs = list(df['Idx'])
    rst = []
    for idx in idxs:
        rst.append(PLIF(idx))
    return np.array(rst)


np.savetxt(os.path.join(base_dir, 'processed_plif.txt'),PLIF_all(),fmt='%d')

