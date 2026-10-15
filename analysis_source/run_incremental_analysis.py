"""Post hoc routine-CT controls and fixed-OOF recorded-group comparisons.

Original plans, cohorts, predictions, models and SHAP records are read only.
All writes are confined to this additive revision directory.
"""
import os,sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,time,platform,warnings,importlib.util,argparse
import numpy as np
import pandas as pd
import scipy,sklearn,joblib
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,cross_val_predict
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from threadpoolctl import threadpool_limits,threadpool_info

REV=Path(__file__).resolve().parents[1];ROOT=REV.parent;OUT=REV/'analysis'
SEED=20260926;B=2000
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
training=load_module('original_training_helpers',ROOT/'code/training.py')
evaluation=load_module('original_evaluation_helpers',ROOT/'code/evaluate_results.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(d,p):d.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def write_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def write_aligned_predictions(frame):
    # Preserve the original three models' probability/threshold tokens verbatim.
    # This also avoids a repeated CSV float parse/write at machine epsilon.
    old=pd.read_csv(ROOT/'analysis/primary_predictions.csv',dtype=str,keep_default_na=False)
    old.loc[:,'node_rads']=old.node_id.map(frame.set_index('node_id').node_rads.astype(int).astype(str))
    new=pd.read_csv(OUT/'new_model_predictions.csv',dtype=str,keep_default_na=False)
    combined=pd.concat([old,new],ignore_index=True)
    combined.to_csv(OUT/'predictions_all_models.csv',index=False,encoding='utf-8-sig')

def fit_controls(frame,specs):
    pred=[];settings=[];grids=[];inner_map=[];overlap=[]
    models=OUT/'new_models';models.mkdir(exist_ok=True)
    for fold in range(1,6):
        tr=frame[frame.outer_fold.ne(fold)].reset_index(drop=True);te=frame[frame.outer_fold.eq(fold)].reset_index(drop=True)
        assert not set(tr.group_id)&set(te.group_id)
        inner=training.grouped_splits(tr,3,SEED+fold)
        for k,(a,b) in enumerate(inner,1):
            assert not set(tr.iloc[a].group_id)&set(tr.iloc[b].group_id)
            for i in b:inner_map.append({'outer_fold':fold,'inner_fold':k,'node_id':tr.iloc[i].node_id,'group_id':tr.iloc[i].group_id,'outcome':int(tr.iloc[i].outcome)})
        overlap.append({'outer_fold':fold,'development_nodes':len(tr),'evaluation_nodes':len(te),'development_groups':tr.group_id.nunique(),'evaluation_groups':te.group_id.nunique(),'outer_group_overlap':0,'inner_group_overlap':0})
        for name,features in specs.items():
            start=time.perf_counter()
            pipe=training.make_pipeline(LogisticRegression(max_iter=5000,solver='lbfgs',random_state=SEED))
            search=GridSearchCV(pipe,{'model__C':training.CS},scoring='neg_log_loss',cv=inner,n_jobs=1,refit=True,error_score='raise')
            search.fit(tr[features],tr.outcome)
            p_inner=cross_val_predict(clone(search.best_estimator_),tr[features],tr.outcome,cv=inner,method='predict_proba',n_jobs=1)[:,1]
            threshold=training.get_threshold(tr.outcome.to_numpy(),p_inner)
            elapsed=time.perf_counter()-start;t0=time.perf_counter();prob=search.predict_proba(te[features])[:,1];inference=time.perf_counter()-t0
            for row,p in zip(te.itertuples(),prob):
                pred.append({'node_id':row.node_id,'group_id':row.group_id,'fold':fold,'y_true':int(row.outcome),'model':name,'probability':float(p),'prediction':int(p>=threshold),'threshold':threshold,'node_rads':int(row.node_rads)})
            settings.append({'model':name,'outer_fold':fold,'features':len(features),'C':search.best_params_['model__C'],'threshold':threshold,'best_inner_log_loss':-search.best_score_,'fit_and_threshold_seconds':elapsed,'inference_seconds':inference,'development_nodes':len(tr),'evaluation_nodes':len(te),'evaluation_auc':roc_auc_score(te.outcome,prob),'evaluation_average_precision':average_precision_score(te.outcome,prob),'analysis_role':'post_hoc_added_after_primary_validation'})
            cv=pd.DataFrame(search.cv_results_);cv['model']=name;cv['outer_fold']=fold
            for r in cv.to_dict('records'):r['params']=json.dumps(r['params']);grids.append(r)
            joblib.dump(search.best_estimator_,models/f'{name}_outer_fold_{fold}.joblib')
            write_csv(pd.DataFrame(pred),OUT/'new_model_predictions.csv')
            print(f'New control {name}: outer fold {fold}/5 complete',flush=True)
    write_csv(pd.DataFrame(settings),OUT/'new_model_fitting_details.csv');write_csv(pd.DataFrame(grids),OUT/'new_model_inner_grid_results.csv')
    write_csv(pd.DataFrame(inner_map),OUT/'inner_evaluation_node_mapping.csv');write_csv(pd.DataFrame(overlap),OUT/'group_overlap_checks.csv')
    coefficients=[];fullgrids=[]
    for name,features in specs.items():
        inner=training.grouped_splits(frame,3,SEED+100)
        search=GridSearchCV(training.make_pipeline(LogisticRegression(max_iter=5000,solver='lbfgs',random_state=SEED)),{'model__C':training.CS},scoring='neg_log_loss',cv=inner,n_jobs=1,refit=True,error_score='raise')
        search.fit(frame[features],frame.outcome);f=search.best_estimator_;est=f.named_steps['model'];sc=f.named_steps['scaler'];imp=f.named_steps['imputer']
        beta=est.coef_[0]/sc.scale_;intercept=float(est.intercept_[0]-np.sum(est.coef_[0]*sc.mean_/sc.scale_))
        coefficients.append({'model':name,'feature':'Intercept','coefficient_original_units':intercept,'standardized_coefficient':float(est.intercept_[0]),'imputation_median':np.nan,'training_mean':np.nan,'training_sd':np.nan,'C':search.best_params_['model__C']})
        for j,c in enumerate(features):coefficients.append({'model':name,'feature':c,'coefficient_original_units':beta[j],'standardized_coefficient':est.coef_[0,j],'imputation_median':imp.statistics_[j],'training_mean':sc.mean_[j],'training_sd':sc.scale_[j],'C':search.best_params_['model__C']})
        assert np.max(np.abs(expit(intercept+imp.transform(frame[features])@beta)-f.predict_proba(frame[features])[:,1]))<1e-10
        joblib.dump(f,models/f'{name}_full_development_model.joblib')
        d=pd.DataFrame(search.cv_results_);d['model']=name
        for r in d.to_dict('records'):r['params']=json.dumps(r['params']);fullgrids.append(r)
    write_csv(pd.DataFrame(coefficients),OUT/'new_full_development_coefficients.csv');write_csv(pd.DataFrame(fullgrids),OUT/'new_full_development_grid_results.csv')
    return pd.DataFrame(pred)

def basic_metrics(y,p,pred,w):
    d=evaluation.metric_values(y,p,pred,w,calibration=False);total=w.sum();pc=np.clip(p,evaluation.EPSILON,1-evaluation.EPSILON)
    d['brier_score']=float(np.dot(w,(p-y)**2)/total)
    d['log_loss']=float(-np.dot(w,y*np.log(pc)+(1-y)*np.log1p(-pc))/total)
    return d

def transitions(frame,prob,pred,threshold):
    comparison_pairs=[('NodeRADS','Combined'),('NodeRADS_RoutineCT','Combined'),('NodeRADS','NodeRADS_RoutineCT'),('RoutineCT','Imaging'),('RoutineCT','Combined'),('Imaging','Combined')]
    subgroups={'All':np.ones(len(frame),bool),'NodeRADS2':frame.node_rads.eq(2).to_numpy(),'NodeRADS3':frame.node_rads.eq(3).to_numpy(),'NodeRADS2or3':frame.node_rads.isin([2,3]).to_numpy()}
    records=[];features=['nc_ct','ce_ct','short_axis_mm','long_axis_mm','nc_iodine','ce_iodine','nc_zeff','ce_zeff','nc_ed','ce_ed','ct_70kev','ct_100kev','casupp_30','casupp_70']
    indicators={}
    for before,after in comparison_pairs:
        y=frame.outcome.to_numpy(int);b=pred[before];a=pred[after]
        flags={'corrected_FN':(y==1)&(b==0)&(a==1),'new_FN':(y==1)&(b==1)&(a==0),'corrected_FP':(y==0)&(b==1)&(a==0),'new_FP':(y==0)&(b==0)&(a==1)}
        indicators[(before,after)]=flags
        for i,row in enumerate(frame.itertuples()):
            changed=next((k for k,v in flags.items() if v[i]),'unchanged_correct' if b[i]==y[i] else 'unchanged_error')
            records.append({'comparison':f'{after} - {before}','before_model':before,'after_model':after,'node_id':row.node_id,'group_id':row.group_id,'fold':int(row.outer_fold),'node_rads':int(row.node_rads),'y_true':int(row.outcome),'probability_before':prob[before][i],'probability_after':prob[after][i],'delta_probability':prob[after][i]-prob[before][i],'threshold_before':threshold[before][i],'threshold_after':threshold[after][i],'prediction_before':int(b[i]),'prediction_after':int(a[i]),'correct_before':int(b[i]==y[i]),'correct_after':int(a[i]==y[i]),'classification_transition':changed,**{f:frame.iloc[i][f] for f in features}})
    records=pd.DataFrame(records);write_csv(records,OUT/'classification_transitions_by_node.csv')
    write_csv(records[~records.classification_transition.str.startswith('unchanged')],OUT/'classification_changed_cases.csv')
    return comparison_pairs,subgroups,indicators

def evaluate_all(frame,allpred,new_names,reuse_bootstrap=False):
    names=['NodeRADS','RoutineCT','NodeRADS_RoutineCT','Imaging','Combined'];node_ids=frame.node_id.tolist();y=frame.outcome.to_numpy(int)
    indexed={m:allpred[allpred.model.eq(m)].set_index('node_id').loc[node_ids] for m in names}
    prob={m:d.probability.to_numpy(float) for m,d in indexed.items()};pred={m:d.prediction.to_numpy(int) for m,d in indexed.items()};threshold={m:d.threshold.to_numpy(float) for m,d in indexed.items()}
    groups=np.array(sorted(frame.group_id.unique()),dtype=str);lookup={g:i for i,g in enumerate(groups)};codes=np.array([lookup[g] for g in frame.group_id]);assert len(groups)==130
    write_csv(pd.DataFrame({'bootstrap_group_index':np.arange(len(groups)),'group_id':groups}),OUT/'bootstrap_group_index.csv')
    pairs,subsets,indicators=transitions(frame,prob,pred,threshold)
    keys=list(evaluation.METRICS);point={m:evaluation.metric_values(y,prob[m],pred[m]) for m in names}
    boot={m:{k:np.full(B,np.nan) for k in keys} for m in names}
    binfo={};bcodes={};bboot={};curves=[]
    for m in names:
        bcodes[m],binfo[m]=evaluation.calibration_bin_specification(prob[m],y,m)
        bboot[m]=np.full((B,len(binfo[m])),np.nan)
        for r in binfo[m]:r['n_source_groups']=int(frame.loc[bcodes[m]==r['bin']-1,'group_id'].nunique())
        fp,tp,th=roc_curve(y,prob[m])
        curves.extend({'model':m,'curve':'ROC','x':float(x),'y':float(z),'threshold':float(t) if np.isfinite(t) else np.nan} for x,z,t in zip(fp,tp,th))
        pr,rec,th=precision_recall_curve(y,prob[m])
        curves.extend({'model':m,'curve':'PR','x':float(x),'y':float(z),'threshold':float(th[i]) if i<len(th) else np.nan} for i,(x,z) in enumerate(zip(rec,pr)))
    oldmetrics=pd.read_csv(ROOT/'analysis/metrics_primary.csv')
    for m in ['NodeRADS','Imaging','Combined']:
        for r in oldmetrics[oldmetrics.model.eq(m)].itertuples():assert abs(point[m][r.metric]-r.estimate)<1e-8,(m,r.metric)
    weights=np.zeros((B,len(groups)),np.int16);rng=np.random.default_rng(SEED)
    if reuse_bootstrap:
        cache=np.load(OUT/'bootstrap_metric_arrays.npz');weights=np.load(OUT/'bootstrap_group_weights.npy')
        assert weights.shape==(B,len(groups))
        for m in names:
            for k in keys:boot[m][k]=cache[f'{m}__{k}'].copy()
    transition_keys=['accuracy_difference','sensitivity_difference','specificity_difference','corrected_FN_rate','new_FN_rate','corrected_FP_rate','new_FP_rate']
    transboot={(a,b,s):{k:np.full(B,np.nan) for k in transition_keys} for a,b in pairs for s in subsets}
    start=time.perf_counter()
    for rep in range(B):
        gw=weights[rep] if reuse_bootstrap else np.bincount(rng.integers(0,len(groups),size=len(groups)),minlength=len(groups))
        weights[rep]=gw;w=gw[codes].astype(float)
        for m in names:
            if not reuse_bootstrap:
                result=evaluation.metric_values(y,prob[m],pred[m],w) if m in new_names else basic_metrics(y,prob[m],pred[m],w)
                for k,val in result.items():boot[m][k][rep]=val
            for j in range(len(binfo[m])):
                mask=bcodes[m]==j;bboot[m][rep,j]=evaluation.safe_divide(np.dot(w[mask],y[mask]),w[mask].sum())
        for a,b in pairs:
            flags=indicators[(a,b)]
            for s,mask in subsets.items():
                n=w[mask].sum();pos=np.dot(w[mask],y[mask]);neg=n-pos
                counts={k:float(np.dot(w[mask],v[mask])) for k,v in flags.items()};d=transboot[(a,b,s)]
                d['accuracy_difference'][rep]=evaluation.safe_divide(counts['corrected_FN']+counts['corrected_FP']-counts['new_FN']-counts['new_FP'],n)
                d['sensitivity_difference'][rep]=evaluation.safe_divide(counts['corrected_FN']-counts['new_FN'],pos)
                d['specificity_difference'][rep]=evaluation.safe_divide(counts['corrected_FP']-counts['new_FP'],neg)
                for k,denom in [('corrected_FN_rate',pos),('new_FN_rate',pos),('corrected_FP_rate',neg),('new_FP_rate',neg)]:d[k][rep]=evaluation.safe_divide(counts[k.replace('_rate','')],denom)
        if (rep+1)%250==0:print(f'Fixed-OOF group bootstrap {rep+1}/{B}; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    np.save(OUT/'bootstrap_group_weights.npy',weights)
    np.savez_compressed(OUT/'bootstrap_metric_arrays.npz',**{f'{m}__{k}':v for m,d in boot.items() for k,v in d.items()})
    rows=oldmetrics.to_dict('records')
    for r in rows:r['analysis_role']='reused_frozen_primary_result'
    for m in new_names:
        for k in keys:
            lo,hi,valid=evaluation.interval(boot[m][k]);rows.append({'model':m,'metric':k,'estimate':point[m][k],'ci_lower':lo,'ci_upper':hi,'bootstrap_valid':valid,'n_nodes':248,'n_source_groups':130,'malignant_nodes':84,'interval_method':'fixed_oof_source_group_percentile_bootstrap','analysis_role':'post_hoc_routine_control'})
    write_csv(pd.DataFrame(rows),OUT/'metrics_all_models.csv')
    binrows=[]
    for m in names:
        for j,r in enumerate(binfo[m]):
            lo,hi,valid=evaluation.interval(bboot[m][:,j]);binrows.append({**r,'observed_ci_lower':lo,'observed_ci_upper':hi,'bootstrap_valid':valid,'bin_definition':'fixed_full_oof_probability_quantile_bins'})
    write_csv(pd.DataFrame(binrows),OUT/'calibration_bins.csv');write_csv(pd.DataFrame(curves),OUT/'ROC_PR_coordinates.csv')
    contrasts=[('NodeRADS_RoutineCT','Combined','spectral_10_added_beyond_routine_plus_NodeRADS'),('NodeRADS','NodeRADS_RoutineCT','routine_4_added_beyond_NodeRADS'),('RoutineCT','Imaging','spectral_10_added_beyond_routine'),('RoutineCT','Combined','NodeRADS_plus_spectral_10_added_beyond_routine'),('NodeRADS','Combined','frozen_primary_comparison'),('Imaging','Combined','frozen_primary_comparison'),('Imaging','NodeRADS','secondary_primary_model_pair')]
    differences=[]
    for a,b,role in contrasts:
        for k in ['roc_auc','average_precision','brier_score','log_loss']:
            lo,hi,valid=evaluation.interval(boot[b][k]-boot[a][k]);differences.append({'comparison':f'{b} - {a}','before_model':a,'after_model':b,'metric':k,'difference':point[b][k]-point[a][k],'ci_lower':lo,'ci_upper':hi,'bootstrap_valid':valid,'analysis_role':role,'n_nodes':248,'n_source_groups':130,'new_confirmatory_P_value':False,'scope':'fixed_OOF_post_hoc_exploratory_for_new_controls'})
    raw=pd.read_csv(ROOT/'analysis/raw_noderads_benchmark.csv');r=raw[raw.benchmark.eq('Combined_minus_RawNodeRADS')].iloc[0]
    differences.append({'comparison':'Combined - RawNodeRADS','before_model':'RawNodeRADS','after_model':'Combined','metric':'roc_auc','difference':r.estimate,'ci_lower':r.ci_lower,'ci_upper':r.ci_upper,'bootstrap_valid':r.bootstrap_valid,'analysis_role':'reused_frozen_raw_ordinal_secondary_comparison','n_nodes':248,'n_source_groups':130,'new_confirmatory_P_value':False,'scope':'reused_original_secondary_interval'})
    write_csv(pd.DataFrame(differences),OUT/'paired_differences.csv')
    transitionrows=[]
    for a,b in pairs:
        flags=indicators[(a,b)]
        for s,mask in subsets.items():
            n=int(mask.sum());pos=int(y[mask].sum());neg=n-pos;counts={k:int(v[mask].sum()) for k,v in flags.items()}
            base={'comparison':f'{b} - {a}','subset':s,'n_nodes':n,'malignant_nodes':pos,'benign_nodes':neg,'n_source_groups':frame.loc[mask,'group_id'].nunique(),**counts,'net_corrected_nodes':counts['corrected_FN']+counts['corrected_FP']-counts['new_FN']-counts['new_FP']}
            pointvals={'accuracy_difference':base['net_corrected_nodes']/n,'sensitivity_difference':evaluation.safe_divide(counts['corrected_FN']-counts['new_FN'],pos),'specificity_difference':evaluation.safe_divide(counts['corrected_FP']-counts['new_FP'],neg),'corrected_FN_rate':evaluation.safe_divide(counts['corrected_FN'],pos),'new_FN_rate':evaluation.safe_divide(counts['new_FN'],pos),'corrected_FP_rate':evaluation.safe_divide(counts['corrected_FP'],neg),'new_FP_rate':evaluation.safe_divide(counts['new_FP'],neg)}
            transitionrows.append({**base,'before_correct_nodes':int((pred[a][mask]==y[mask]).sum()),'after_correct_nodes':int((pred[b][mask]==y[mask]).sum()),**pointvals})
            for k,v in pointvals.items():
                lo,hi,valid=evaluation.interval(transboot[(a,b,s)][k]);base[k+'_ci_lower']=lo;base[k+'_ci_upper']=hi;base[k+'_bootstrap_valid']=valid
            transitionrows[-1].update(base)
    write_csv(pd.DataFrame(transitionrows),OUT/'classification_transition_summary.csv')
    examples=json.loads((ROOT/'analysis/shap_examples.json').read_text(encoding='utf-8'))['examples'];case_rows=[]
    for ex in examples:
        n=node_ids.index(ex['node_id'])
        for m in names:
            case_rows.append({**ex,'case_description':'moderate_probability_selected_near_0_5_not_decision_boundary','evaluated_model':m,'probability':prob[m][n],'actual_outer_threshold':threshold[m][n],'predicted_class':int(pred[m][n]),'recorded_class':int(y[n]),'classification_correct':bool(pred[m][n]==y[n]),'probability_minus_threshold':prob[m][n]-threshold[m][n],'change_from_NodeRADS':int(pred[m][n]==y[n])-int(pred['NodeRADS'][n]==y[n])})
    write_csv(pd.DataFrame(case_rows),OUT/'shap_examples_actual_thresholds.csv')
    report={'models':names,'new_models':list(new_names),'n_nodes':248,'n_groups':130,'malignant_nodes':84,'benign_nodes':164,'new_model_outer_fits':10,'new_model_full_development_fits':2,'old_models_retrained':0,'SHAP_values_recomputed':False,'bootstrap_replicates':B,'bootstrap_seed':SEED,'paired_shared_group_draws':True,'no_new_P_or_Holm_family':True,'post_hoc_added_after_primary_validation':True,'subgroups':{s:{'n':int(m.sum()),'malignant':int(y[m].sum()),'benign':int((1-y[m]).sum())} for s,m in subsets.items()},'calibration_bins':{m:len(binfo[m]) for m in names},'bootstrap_elapsed_seconds':time.perf_counter()-start,'invalid_bootstrap_metrics':{m:{k:int((~np.isfinite(boot[m][k])).sum()) for k in keys} for m in new_names},'runtime_threadpools':threadpool_info()}
    write_json(OUT/'analysis_completion_QA.json',report)
    return pd.DataFrame(rows),pd.DataFrame(differences),pd.DataFrame(transitionrows),pd.DataFrame(case_rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-fitted-and-bootstrap',action='store_true');args=parser.parse_args()
    plan=json.loads((OUT/'incremental_analysis_plan.json').read_text(encoding='utf-8'));before=plan['source_files_SHA256_before']
    assert all(sha(p)==h for p,h in before.items())
    frame=pd.read_csv(ROOT/'analysis/cohort_primary.csv',dtype={'node_id':str,'group_id':str})
    assert len(frame)==248 and frame.group_id.nunique()==130 and frame.outcome.sum()==84 and frame.groupby('group_id').outer_fold.nunique().max()==1
    assert set(plan['routine_feature_subset'])|set(plan['spectral_feature_subset'])==set(training.FEATURES)
    with threadpool_limits(limits=1),joblib.parallel_backend('threading',n_jobs=1):
        if args.reuse_fitted_and_bootstrap:
            allpred=pd.read_csv(OUT/'predictions_all_models.csv',dtype={'node_id':str,'group_id':str})
        else:
            new=fit_controls(frame,plan['new_models']);old=pd.read_csv(ROOT/'analysis/primary_predictions.csv',dtype={'node_id':str,'group_id':str})
            old=old.merge(frame[['node_id','node_rads']],on='node_id',validate='many_to_one');allpred=pd.concat([old,new],ignore_index=True)
        assert len(allpred)==1240 and not allpred.duplicated(['node_id','model']).any()
        assert np.array_equal((allpred.probability>=allpred.threshold).astype(int),allpred.prediction)
        write_aligned_predictions(frame)
        metrics,differences,transitions,cases=evaluate_all(frame,allpred,plan['new_models'],args.reuse_fitted_and_bootstrap)
    after={p:sha(p) for p in before};assert before==after
    write_json(OUT/'source_hashes_after.json',after)
    notes=['# 新增探索性常规CT对照与DECT参数增量分析','', '该分析于2026年10月10日在主要验证完成后追加。沿用248枚节点、130个源分组及原外层分配，不属于新的未见验证队列。原始文件、主要模型与其预测未改动。', '', '新增RoutineCT仅含非增强/增强CT值和短轴/长轴四项，NodeRADS_RoutineCT再加Node-RADS。其余十项为两期碘浓度、有效原子序数、电子密度、70/100 keV及两种钙抑制设置。常规CT只是同批扫描的输入子集。', '', '所有新增模型使用原三内层分组、训练折插补和标准化、相同C网格及log loss调参，阈值来自内层OOF Youden。亚组不调整阈值。2,000次源分组bootstrap保持所有预测固定，没有新增确认性P值或调整原Holm比较。', '', '## 模型增量','']
    for c in ['Combined - NodeRADS_RoutineCT','NodeRADS_RoutineCT - NodeRADS','Imaging - RoutineCT']:
        for k in ['roc_auc','average_precision','brier_score']:
            z=differences[(differences.comparison==c)&(differences.metric==k)].iloc[0];notes.append(f'{c}，{k}差值 {z.difference:.4f}（95% CI {z.ci_lower:.4f}至{z.ci_upper:.4f}）。')
        notes.append('')
    notes+=['## Node-RADS 2与3级分类变化','']
    for c in ['Combined - NodeRADS','Combined - NodeRADS_RoutineCT']:
        for s in ['NodeRADS2','NodeRADS3','NodeRADS2or3']:
            z=transitions[(transitions.comparison==c)&(transitions.subset==s)].iloc[0]
            notes.append(f'{c}，{s}：{z.n_nodes}枚，纠正FN {z.corrected_FN}、新增FN {z.new_FN}、纠正FP {z.corrected_FP}、新增FP {z.new_FP}，净增加正确分类 {z.net_corrected_nodes}枚。准确率差 {z.accuracy_difference:.4f}（95% CI {z.accuracy_difference_ci_lower:.4f}至{z.accuracy_difference_ci_upper:.4f}）。')
        notes.append('')
    notes+=['分类纠正指与记录结局更一致，不能解释为治疗获益。节点病理标记与源分组医院身份的原有限制仍存在。源分组bootstrap保留组内关联，未采用假定节点独立的McNemar检验。', '', '旧SHAP案例继续使用原值，现称中等预测概率案例。接近0.5不等同于各外层模型实际决策阈值。其实际阈值、类别和分类正误见shap_examples_actual_thresholds.csv。', '', f'保护的{len(before)}个原文件SHA256全部保持。新增输出仅在本目录。']
    (OUT/'incremental_analysis_summary.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print('COMPLETE: protected source hashes unchanged',flush=True)

if __name__=='__main__':main()
