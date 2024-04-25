import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
import argparse
import datetime
from numpy import ceil
import multiprocessing as mp

DIR = os.path.dirname(os.path.abspath(__file__))
inputfile = os.path.join(DIR,'smiles_pdbqt.csv')
outfile = os.path.join(DIR,"smiles_dock_out.csv")
outlogfile = os.path.join(DIR,"generate_dock_out.log")

vina_path = '/home/hy/Softwares/Bioinformatics/autodock/vina_1.2.3_linux_x86_64'
config_path = os.path.join(DIR, 'config.txt')

parser = argparse.ArgumentParser(description='vina docking')
parser.add_argument("-i", "--inputfile", type=str, default=inputfile,
                    help="smiles input file, should not include a head")
parser.add_argument("-v", "--vina", type=str, default=vina_path,
                    help="vina path")
parser.add_argument("-c", "--config", type=str, default=config_path,
                    help="config.txt path")
parser.add_argument("-n", "--cores", type=int, default=8,
                    help="number of cores used")
parser.add_argument("-o", "--outfile", type=str, default=outfile,
                    help="out file")
parser.add_argument("-m", "--multi", type=bool, default=False,
                    help="if multi-cores")
args = parser.parse_args()



if args.multi:
    save_dir =os.path.join(DIR,'dock_multi_cores')
else:
    save_dir =os.path.join(DIR,'dock_single_core')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def generate_dock_out(df, save_dir=save_dir):
    config_dir = os.path.join(save_dir,'config')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    dock_out_dir = os.path.join(save_dir,'dock_out')
    if not os.path.exists(dock_out_dir):
        os.mkdir(dock_out_dir)

    with open(os.path.join(save_dir,'generate_dock_out.log'), 'w') as log:
        
        dock_out_ls = []
        config_ls = []
        affnity_ls = []
        for index,row in df.iterrows():
            pdbqt_path = row['Pdbqt']
            config_path = '{}/config{}.txt'.format(config_dir,index)
            dock_out_path = '{}/dock_out{}.pdbqt'.format(dock_out_dir, index)
            log_path = '{}/log_temp.txt'.format(save_dir)
            if pdbqt_path.startswith('/home'):
                # 1. write config.txt
                with open(config_path,'w') as f:
                    f.write('receptor = {}/receptor.pdbqt\n'.format(DIR))
                    f.write('ligand = {}\n'.format(pdbqt_path))
                    f.write('center_x = 121.4\n')
                    f.write('center_y = 113.5\n')
                    f.write('center_z = 79.8\n')
                    f.write('size_x = 22.5\n')
                    f.write('size_y = 23.3\n')
                    f.write('size_z = 18.8\n')
                    f.write('out = {}\n'.format(dock_out_path))
                # 2. run
                os.system('cd {} ;{} --config {} --exhaustiveness 1 > {} 2>&1'.format(DIR, args.vina, config_path, log_path))
                # 3. read log and record 
                with open(log_path,'r') as log_temp:
                    lines = log_temp.readlines()
                    print(index)
                    log.write('>mol{}\n'.format(index))
                    for line in lines:
                        log.write(line)
                dock_out_ls.append(dock_out_path)
                config_ls.append(config_path)
                affnity_ls.append(open(dock_out_path,'r').readlines()[1].split()[3])
            else:
                dock_out_ls.append('Error')
                config_ls.append('Error')
                affnity_ls.append('Error')
                log.write('>mol{}\n'.format(index))
                log.write('Error\n')
    df.insert(df.shape[1],'Config',config_ls)
    df.insert(df.shape[1],'DockOut',dock_out_ls)
    df.insert(df.shape[1],'Affinity',affnity_ls)
    df.to_csv(args.outfile,header=True,index=False,sep='\t')
    os.system('rm {}'.format(log_path))

def generate_dock_out_multicores(name, param):
    df = param
    son_dir = os.path.join(save_dir, name)
    if not os.path.exists(son_dir):
        os.mkdir(son_dir) 

    outfile = os.path.join(son_dir,'all.csv')

    dock_out_dir = os.path.join(son_dir,'dock_out')
    if not os.path.exists(dock_out_dir):
        os.mkdir(dock_out_dir)

    config_dir = os.path.join(son_dir,'config')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    with open(os.path.join(son_dir,'generate_dock_out.log'), 'w') as log:
        dock_out_ls = []
        config_ls = []
        affnity_ls = []
        for index,row in df.iterrows():
            pdbqt_path = row['Pdbqt']
            config_path = '{}/config{}.txt'.format(config_dir, index)
            dock_out_path = '{}/dock_out{}.pdbqt'.format(dock_out_dir, index)
            log_path = '{}/log_temp.txt'.format(son_dir)
            if pdbqt_path.startswith('/home'):
                # 1. write config.txt
                with open(config_path,'w') as f:
                    f.write('receptor = {}/receptor.pdbqt\n'.format(DIR))
                    f.write('ligand = {}\n'.format(pdbqt_path))
                    f.write('center_x = 120.2\n')
                    f.write('center_y = 112.7\n')
                    f.write('center_z = 80.5\n')
                    f.write('size_x = 12.1\n')
                    f.write('size_y = 12.0\n')
                    f.write('size_z = 12.9\n')
                    f.write('out = {}\n'.format(dock_out_path))
                os.system('cd {} ;{} --config {} --exhaustiveness 1 > {}'.format(DIR, args.vina, config_path, log_path))
                with open(log_path,'r') as log_temp:
                    lines = log_temp.readlines()
                    log.write('>mol{}\n'.format(index))
                    for line in lines:
                        log.write(line)
                dock_out_ls.append(dock_out_path)
                config_ls.append(config_path)
                affnity_ls.append(open(dock_out_path,'r').readlines()[1].split()[3])
            else:
                dock_out_ls.append('Error')
                config_ls.append('Error')
                affnity_ls.append('Error')
                log.write('>mol{}\n'.format(index))
                log.write('Error\n')
    df.insert(df.shape[1],'Config',config_ls)
    df.insert(df.shape[1],'DockOut',dock_out_ls)
    df.insert(df.shape[1],'Affinity',affnity_ls)
    df.to_csv(outfile,header=True,index=False,sep='\t')
    os.system('rm {}'.format(log_path))


def multi_tasks(df, outfile=args.outfile,logpath=outlogfile,num_cores=args.cores):
    
    print('read ok!')
    num = len(df)
    print('mols:',num)
    print("computer has: " + str(int(mp.cpu_count())) + " cores")

    start_t = datetime.datetime.now()
    pool = mp.Pool(num_cores)
    inteval = int(ceil(num/num_cores))
    ls = list(range(0, num, inteval))
    dic={}
    for i in range(len(ls)):
        if i!=(len(ls)-1):
            dic['task'+str(i)] = df[ls[i]:ls[i+1]]
        else:
            dic['task'+str(i)] = df[ls[i]:]
    print(ls)
    print(dic)

    results = [pool.apply_async(generate_dock_out_multicores, args=(name, param)) for name, param in dic.items()]
    results = [p.get() for p in results]
    print(results)


    rst_csv = pd.DataFrame()
    rst_log  = []
    for name in dic.keys():
        dir = os.path.join(DIR,'dock_multi_cores/{}'.format(name))
        csv = dir + '/all.csv'
        log = dir + '/generate_pdbqt.log'

        df = pd.read_csv(csv,sep='\t',header=0)
        rst_csv = pd.concat([rst_csv, df])
        lines = open(log,'r').readlines()
        rst_log += lines
    rst_csv.to_csv(outfile,header=True,sep='\t',index=False)
    with open(logpath,'w') as f:
        for line in rst_log:
            f.write(line)

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("total cost: " + "{:.2f}".format(elapsed_sec) + " seconds")


df = pd.read_csv(args.inputfile,sep='\t',header=0)
if not args.multi:
    generate_dock_out(df)
else:
    multi_tasks(df)