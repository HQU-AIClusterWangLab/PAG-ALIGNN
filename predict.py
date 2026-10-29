"""Run the selected PAG-ALIGNN ensemble on packaged or custom inputs."""
from pathlib import Path
import argparse,json,os,runpy,sys,tempfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,default=ROOT/'work/predictions.csv');p.add_argument('--structures',type=Path,default=ROOT/'data/inference/structures.json');p.add_argument('--aux',type=Path,default=ROOT/'data/inference/aux.csv');p.add_argument('--cat',type=Path,default=ROOT/'data/inference/catboost.csv');args=p.parse_args()
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True);structures=args.structures.resolve();aux=args.aux.resolve();cat=args.cat.resolve()
    work=Path(tempfile.mkdtemp(prefix='pag-inference-',dir=output.parent))
    if args.smoke:
        values=json.loads(structures.read_text(encoding='utf8'));uids={'2LiMnCl3-1','4FIn-2'};subset=[v for v in values if v.get('uid',v.get('id')) in uids]
        assert len(subset)==2,'Smoke mode expects packaged inputs.'
        structures=work.parent/(work.name+'-samples.json');structures.write_text(json.dumps(subset),encoding='utf8')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/scripts'))
    sys.argv=['predict_routed_polarizability.py','--bundle-dir','models','--structures-json',str(structures),'--alignn-aux-csv',str(aux),'--cat-descriptors',str(cat),'--output',str(output),'--work-dir',str(work),'--device',args.device,'--eta-model-label','anisotropy','--inplane-model-label','inplane']
    runpy.run_path(str(ROOT/'code/scripts/predict_routed_polarizability.py'),run_name='__main__')
if __name__=='__main__':main()
