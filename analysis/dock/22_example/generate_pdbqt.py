import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
import argparse
import datetime

DIR = os.path.dirname(os.path.abspath(__file__))
print(DIR)

inputfile = os.path.join(DIR+'/smiles_mol2.csv')
outfile = os.path.join(DIR+'/smiles_pdbqt.csv')
prepare_ligand4_path = '/home/hy/Softwares/Bioinformatics/autodock/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'

parser = argparse.ArgumentParser(description='rdkit descriptors')
parser.add_argument("-i", "--inputfile", type=str, default=inputfile,
                    help="smiles input file, should not include a head")
parser.add_argument("-p", "--pl4path", type=str, default=prepare_ligand4_path,
                    help="prepare_ligand4.py path")
parser.add_argument("-o", "--outfile", type=str, default=outfile,
                    help="out file")
args = parser.parse_args()

save_dir =os.path.join(DIR,'pdbqt')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

import shutil
def generate_mol2(df,save_dir=save_dir):
    ls = []
    with open(os.path.join(save_dir,'../generate_pdbqt.log'), 'w') as log:
        for index,row in df.iterrows():
            print(index)
            pdbqt_path = '{}/ligand{}.pdbqt'.format(save_dir, index)
            log_path = '{}/../log_temp.txt'.format(save_dir)
            mol2_path = row['Mol2']
            if mol2_path !='Error' and mol2_path != 'Warning':
                shutil.copy(mol2_path,save_dir+'/../ligand.mol2')
                os.system('cd {}; {} -l ligand.mol2 -o {} > {} 2>&1'.format(DIR, args.pl4path, pdbqt_path, log_path))
                with open(log_path,'r') as log_temp:
                    lines = log_temp.readlines()
                    for line in lines:
                        log.write(line)
                    if lines[0] == 'setting PYTHONHOME environment\n' and len(lines) == 1:
                        ls.append(pdbqt_path)
                    else:
                        ls.append(lines[1].split()[0])
            else:
                ls.append('Error')
    os.system('rm {}'.format(save_dir+'/../ligand.mol2'))
    os.system('rm {}'.format(log_path))
    df.insert(df.shape[1],'Pdbqt',ls)
    df.to_csv(args.outfile,header=True,index=False,sep='\t')

df = pd.read_csv(args.inputfile,sep='\t',header=0)
generate_mol2(df)











