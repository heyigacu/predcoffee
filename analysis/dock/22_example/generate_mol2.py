import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os
import argparse
import subprocess


DIR = os.path.dirname(os.path.abspath(__file__))
inputfile = os.path.join(DIR+'/smiles.csv')
outfile_name = os.path.join(DIR+'/smiles_mol2.csv')

parser = argparse.ArgumentParser(description='rdkit descriptors')
parser.add_argument("-i", "--inputfile", type=str, default=inputfile,
                    help="smiles input file, should include a head")
parser.add_argument("-o", "--outfile", type=str, default=os.path.join(DIR,outfile_name),
                    help="out descriptors file")
args = parser.parse_args()


save_dir =os.path.join(DIR,'mol2s')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


ls =[]
def generate_mol2(smileses,mol2s_dir=save_dir):
    with open(os.path.join(mol2s_dir,'../generate_mol2.log'), 'w') as log:
        for i,smiles in enumerate(smileses):
            lig_path = '{}/ligand{}.mol2'.format(mol2s_dir, i)
            log_path = '{}/../log_temp.txt'.format(mol2s_dir)
            os.system('obabel -:"{}" -omol2 -O {} --gen3d > {} 2>&1'.format(smiles,lig_path,log_path))
            with open(log_path,'r') as log_temp:
                lines = log_temp.readlines()
                print(lines)
                for line in lines:
                    log.write(line)
                if lines[0] == '1 molecule converted\n':
                    ls.append(lig_path)
                else:
                    ls.append(lines[1].split()[3] )

    os.system('rm {}'.format(log_path))
    return ls

df = pd.read_csv(args.inputfile,header=0,sep='\t')
mol2_ls = generate_mol2(list(df['Smiles']))
df.insert(df.shape[1],'Mol2',mol2_ls)
df.to_csv(args.outfile,sep='\t',index=False,header=True)
