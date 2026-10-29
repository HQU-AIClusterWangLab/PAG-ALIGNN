"""Verify checksums and reproduce headline metrics using Python's standard library."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, math, statistics as st
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9),(a,b)
def tensor(data):
    weights={'xx':1,'yy':1,'zz':1,'xy':2,'xz':2,'yz':2};f=[];t=[];r=[];c={k:[] for k in weights}
    for row in data:
        errors={k:float(row['prediction_'+k])-float(row['target_'+k]) for k in weights}
        for k in c:c[k].append(errors[k])
        norm=math.sqrt(sum(w*float(row['target_'+k])**2 for k,w in weights.items()))
        e=math.sqrt(sum(w*errors[k]**2 for k,w in weights.items()))
        f.append(e);r.append(e/max(norm,1e-12));t.append(sum(errors[k] for k in ['xx','yy','zz']))
    out={'n_samples':len(data),'frobenius_absolute_error':st.mean(f),'frobenius_rmse':math.sqrt(st.mean(x*x for x in f)),'relative_frobenius_error':st.mean(r),'trace_mae':st.mean(map(abs,t)),'trace_rmse':math.sqrt(st.mean(x*x for x in t)),'frobenius_median':st.median(f),'relative_frobenius_median':st.median(r)}
    for k,e in c.items():out[k+'_mae']=st.mean(map(abs,e));out[k+'_rmse']=math.sqrt(st.mean(x*x for x in e))
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);p.add_argument('--skip-hashes',action='store_true');args=p.parse_args();n_hash=0
    if (ROOT/'MANIFEST.json').exists() and not args.skip_hashes:
        for item in read('MANIFEST.json')['files']:
            path=ROOT/item['file'];assert path.is_file(),path;assert sha(path)==item['sha256'],path;n_hash+=1
    for path in (ROOT/'code').rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    a=tensor(rows('supplementary_data/D08_tensor_predictions.csv'));expected=read('results/tensor/metrics.json')
    for k,v in a.items():
        if k in expected:close(v,expected[k])
    base=tensor(rows('results/baselines/six_component/predictions.csv'));expected=read('results/baselines/six_component/metrics.json')
    for k,v in base.items():
        if k in expected:close(v,expected[k])
    eta=rows('supplementary_data/D09_eta_predictions.csv');errors=[float(x['prediction_eta_parallel'])-float(x['target_eta_parallel']) for x in eta]
    mae=st.mean(map(abs,errors));rmse=math.sqrt(st.mean(e*e for e in errors));close(mae,.07885892849921353);close(rmse,.1447112214677762)
    for row in eta:close(float(row['prediction_eta_parallel']),min(1,max(0,float(row['raw_prediction_eta_parallel']))))
    names=['single','single_tail','single_under','single_tail_under','multitask','multitask_tail','multitask_under','multitask_tail_under']
    for name in names:assert len(rows('results/ablation/'+name+'/predictions.csv'))==2001
    fusion=rows('supplementary_data/D14_classifier_oof.csv');cm={'tn':0,'fp':0,'fn':0,'tp':0}
    for row in fusion:
        prob=.96*float(row['p_catboost'])+.04*float(row['p_alignn']);close(prob,float(row['p_fused']))
        pred=int(prob>=.06761944);assert pred==int(row['prediction_fused']);cm[['tn','fp','fn','tp'][2*int(row['target'])+pred]]+=1
    assert len(fusion)==1998 and cm=={'tn':1587,'fp':215,'fn':17,'tp':179},cm
    routes=rows('supplementary_data/D01_routed_predictions.csv');assert len(routes)==3351
    assert sum(x['prediction_branch']=='eta_model' for x in routes)==2515
    for row in routes:
        prob=.96*float(row['gate_p_cat'])+.04*float(row['gate_p_alignn']);close(prob,float(row['gate_p_final']));assert int(row['gate_high_alpha'])==int(prob>=.06761944)
    candidates=rows('supplementary_data/D04_tensor_candidates.csv')
    raw=rows('supplementary_data/D04_raw_screen_17.csv')
    assert len(candidates)==16 and len({r['sample_id'] for r in candidates})==16
    assert len(raw)==17
    assert {r['sample_id'] for r in raw}-{r['sample_id'] for r in candidates}=={'2ClPSnO3-1'}
    assert [int(r['manuscript_rank']) for r in candidates]==list(range(1,17))
    si=(ROOT/'docs/Supporting_Information_source.tex').read_text(encoding='utf-8')
    table=si.split('\\label{tab:tensor-route-ranking}',1)[1].split('\\end{table}',1)[0].replace('\\allowbreak ','')
    positions=[table.index(r['sample_id']) for r in candidates]
    assert positions==sorted(positions)
    assert len(rows('data/publication_figures/F09_colorbar_context.csv'))==314
    publication_labels=['xx','yy','zz','xy','xz','yz']
    paired=rows('data/publication_figures/F07_unified_paired_errors.csv')
    assert len(paired)==1803
    close(st.mean(float(r['error_pag']) for r in paired),a['frobenius_absolute_error'])
    ef_aligned=tensor(rows('data/publication_figures/F07_unified_eatgnn_aligned.csv'))
    close(ef_aligned['frobenius_absolute_error'],read('results/baselines/eatgnn/metrics.json')['frobenius_absolute_error'])
    for path in (ROOT/'code/publication_figures').glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    configs=read('configs/experiment_index.json')
    for key,item in configs.items():
        config=read(item['config']);mapping=read(item['input']+'/structures.json');assert len(mapping)==item['samples'],key
        for h in set(mapping.values()):assert (ROOT/'data/structure_pool'/f'{h}.vasp').is_file(),(key,h)
        if config.get('sample_weight_csv'):assert (ROOT/config['sample_weight_csv']).is_file(),key
        assert (ROOT/item['model']/'best_model.pt').is_file(),key
    eat=[]
    for i in range(1,6):eat+=read(f'results/baselines/eatgnn/fold_{i:02d}/test_predictions_best.json')
    assert len(eat)==1803;ef=[];et=[]
    for row in eat:
        d=[[row['pred_tensor'][i][j]-row['target_tensor'][i][j] for j in range(3)] for i in range(3)];ef.append(math.sqrt(sum(x*x for row in d for x in row)));et.append(sum(d[i][i] for i in range(3)))
    expected=read('results/baselines/eatgnn/metrics.json')
    for k,v in {'frobenius_absolute_error':st.mean(ef),'frobenius_rmse':math.sqrt(st.mean(x*x for x in ef)),'trace_mae':st.mean(map(abs,et)),'trace_rmse':math.sqrt(st.mean(x*x for x in et))}.items():close(v,expected[k])
    result={'status':'PASS','checksummed_files':n_hash,'tensor':a,'eta':{'n':len(eta),'mae':mae,'rmse':rmse},'routes':3351,'candidates':16,'raw_screen_candidates':17,'publication_figures':10,'synchronized_date':'2026-10-27','ablation_conditions':8,'training_entries':len(configs),'eatgnn_samples':len(eat)}
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
