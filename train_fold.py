"""Reconstruct exact archived fold inputs and run the recorded training settings."""
from pathlib import Path
import argparse,csv,json,os,shutil,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def main():
    index=json.loads((ROOT/'configs/experiment_index.json').read_text(encoding='utf8'));p=argparse.ArgumentParser();p.add_argument('--list',action='store_true');p.add_argument('--experiment',default='main/eta/fold_01');p.add_argument('--output',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--epochs',type=int);a=p.parse_args()
    if a.list:print('\n'.join(index));return
    item=index[a.experiment];out=(a.output or ROOT/'work/training'/a.experiment).resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Choose a new empty output directory.')
    data=out/'input';data.mkdir(parents=True,exist_ok=True);source=ROOT/item['input'];mapping=json.loads((source/'structures.json').read_text())
    for name,h in mapping.items():
        target=data/name
        if target.parent.resolve()!=data.resolve():raise ValueError('Unexpected structure filename')
        shutil.copyfile(ROOT/'data/structure_pool'/f'{h}.vasp',target)
    for name in ['id_prop.csv','ids_train_val_test.json','sample_weights.csv','stratify_labels.csv']:
        if (source/name).exists():shutil.copyfile(source/name,data/name)
    config=json.loads((ROOT/item['config']).read_text(encoding='utf8'));config['output_dir']=str(out/'run')
    if config.get('sample_weight_csv'):config['sample_weight_csv']=str(data/'sample_weights.csv')
    if a.epochs:config['epochs']=a.epochs
    config_path=out/'config.json';config_path.write_text(json.dumps(config,indent=2),encoding='utf8');features=json.loads((ROOT/item['model']/'extra_feature_names.json').read_text())
    with (ROOT/item['features']).open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);records=list(reader);uid='uid' if 'uid' in reader.fieldnames else 'id'
    assert set(mapping).issubset({r[uid] for r in records}), 'Missing training features'
    feature_path=out/'features.csv'
    with feature_path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['uid']+features);writer.writeheader();writer.writerows({'uid':r[uid],**{k:r[k] for k in features}} for r in records if r[uid] in mapping)
    command=[sys.executable,'-m','alignn.train_alignn','--root_dir',str(data),'--config_name',str(config_path),'--output_dir',str(out/'run'),'--extra_features_csv',str(feature_path),'--extra_features_id_key','uid']
    (out/'command.json').write_text(json.dumps(command,indent=2),encoding='utf8');print('Prepared',a.experiment,'samples',len(mapping),'features',len(features));print(' '.join(command))
    if not a.prepare_only:
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(ROOT/'code')+os.pathsep+env.get('PYTHONPATH','');subprocess.run(command,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
