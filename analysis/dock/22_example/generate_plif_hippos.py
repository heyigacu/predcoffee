import os
import argparse
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
infile = os.path.join(DIR,'smiles_dock_out.csv')
outfile = os.path.join(DIR,'smiles_plif_hippos.csv')
outlogfile = os.path.join(DIR,"generate_plif_hippos.log")
config_path = os.path.join(DIR, 'config.txt')

parser = argparse.ArgumentParser(description='vina docking')
parser.add_argument("-i", "--inputfile", type=str, default=infile,
                    help="smiles input file, should not include a head")
parser.add_argument("-c", "--config", type=str, default=config_path,
                    help="config.txt path")
parser.add_argument("-n", "--cores", type=int, default=16,
                    help="number of cores used")
parser.add_argument("-o", "--outfile", type=str, default=outfile,
                    help="out file")
parser.add_argument("-m", "--multi", type=bool, default=True,
                    help="if multi-cores")
args = parser.parse_args()

save_dir = os.path.join(DIR,'plif_hippos')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)



def generate_plif_hippos(df, save_dir=save_dir):
    # please do below in pymol
    # select 7A, byres ligand around 7
    # iterate 7A, print(resi), print(resi)
    # iterate 7A, f=open("/home/hy/Documents/Project/odor/coffee/analysis/dock/37/resin_7A.txt", "a").write(resn + str(resi) +"\n")
    # iterate 7A, f=open("/home/hy/Documents/Project/odor/coffee/analysis/dock/37/resi_7A.txt", "a").write(str(resi) + "\n")
    # check file status
    resni_string = ''
    df_ni = pd.read_csv('{}/resin_7A.txt'.format(DIR),header=None,sep='\t')
    df_ni.columns = ['res']
    df_ni = df_ni.drop_duplicates(['res'],keep='first')
    for line in list(df_ni['res']):
        resni_string += (line.strip()+' ')

    resi_string = ''
    df_i = pd.read_csv('{}/resi_7A.txt'.format(DIR),header=None,sep='\t')
    df_i.columns = ['res']
    df_i = df_i.drop_duplicates(['res'],keep='first')
    for line in list(df_i['res']) :
        resi_string += (str(int(line)-3)+' ')

    cfg_dir = os.path.join(save_dir,'config')
    if not os.path.exists(cfg_dir):
        os.mkdir(cfg_dir)
    plif_dir = os.path.join(save_dir,'plif')
    if not os.path.exists(plif_dir):
        os.mkdir(plif_dir)    
    log_dir = os.path.join(save_dir,'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)    
    with open(outlogfile, 'w') as log:
        plif_config_ls = []
        plif_out_ls = []
        features = []
        for index,row in df.iterrows():
            config_path = row['Config']
            plif_cfg = '{}/config-vina-na-notc-{}.txt'.format(cfg_dir,index)
            plif_out = '{}/vina_notc_ifp{}.csv'.format(plif_dir,index)
            log_path = '{}/log_temp.txt'.format(save_dir)
            if config_path.startswith('/home'):
                # 1. write config.txt
                with open(plif_cfg,'w') as f:
                    f.write('docking_method    vina\n')
                    f.write('docking_conf      {}\n'.format(config_path))
                    f.write('residue_name      {}\n'.format(resni_string))
                    f.write('\n')
                    f.write('residue_number      {}\n'.format(resi_string))
                    f.write('full_outfile       {}\n'.format(plif_out))
                    f.write('logfile      {}/vina_notc{}.log \n'.format(log_dir,index))
                # 2. run
                os.system("hippos {}/config-vina-na-notc-{}.txt > {}".format(cfg_dir, index, log_path))
                # 3. read log and record 
                with open(log_path,'r') as log_temp:
                    lines = log_temp.readlines()
                    log.write('>mol{}\n'.format(index))
                    for line in lines:
                        log.write(line)
                plif_out_ls.append(plif_out)
                plif_config_ls.append(plif_cfg)
                with open(plif_out,'r') as f:
                    line = f.readlines()[0].strip().split()
                features.append(line[2])
            else:
                plif_out_ls.append('Error')
                plif_config_ls.append('Error')
                features.append('Error')
                log.write('>mol{}\n'.format(index))
                log.write('Error\n')
    df.insert(df.shape[1],'PlifHipposConfig',plif_config_ls)
    df.insert(df.shape[1],'PlifHipposOut',plif_out_ls)
    df.insert(df.shape[1],'PlifHippos',features)
    df.to_csv(args.outfile,header=True,index=False,sep='\t')
    os.system('rm {}'.format(log_path))

df=pd.read_csv(args.inputfile,sep='\t',header=0)
generate_plif_hippos(df)