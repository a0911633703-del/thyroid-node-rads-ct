"""Frozen grouped reanalysis; outer predictions never enter model/threshold selection."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
from hashlib import sha256
from datetime import datetime
import argparse,json,time,warnings,platform
import numpy as np
import pandas as pd
import scipy,sklearn,joblib
from scipy.special import expit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
AN=ROOT/'analysis'; CODE=ROOT/'code'
AN.mkdir(exist_ok=True); (AN/'models').mkdir(exist_ok=True)
SOURCE=ROOT/'source_measurements'/'node_measurements_clean_v1.csv'
REFERENCE=SOURCE.parent/'reference_and_legacy_variables_v1.csv'
SEED=20260926
FEATURES=['nc_ct','ce_ct','nc_iodine','ce_iodine','nc_zeff','ce_zeff','nc_ed','ce_ed',
          'ct_70kev','ct_100kev','casupp_30','casupp_70','short_axis_mm','long_axis_mm']
CS=[0.01,0.1,1.0,10.0,100.0]
METHODS={
 'seed':SEED,'outer_folds':5,'inner_folds':3,'primary_size_criterion':'short_axis_mm > 3',
 'primary_features':FEATURES,'main_models':['NodeRADS','Imaging','Combined'],
 'ridge_C_grid':CS,'main_tuning_metric':'negative log loss',
 'threshold':'maximum Youden index on inner out-of-fold training probabilities; highest threshold for ties',
 'preprocessing':'fold-specific median imputation and standard scaling',
 'grouping':'source-recorded patient identifiers, not independently verified against hospital records',
 'outer_assignment':'generated once in full 254-row source table, kept across prespecified cohorts',
 'bootstrap':'2000 source-group resamples, fixed OOF predictions; two primary AUC contrasts with Holm adjustment',
 'sensitivity_scenarios':['all_254','legacy_249','reference_augmented','normalized_CT','exclude_group_76'],
 'exploratory_algorithms':11,'exploratory_tuning_metric':'ROC AUC','exploratory_primary_summary':'mean outer-fold ROC AUC; no winner chosen from outer results',
 'SVM_score':'decision_function, without ungrouped built-in probability calibration',
 'shap':'linear interventional, all corresponding outer-training nodes as background, log-odds scale',
 'SHAP_examples':'held-out node closest to 0.5 within each recorded outcome class',
 'DCA_threshold_range':[0.05,0.5],'DCA_interpretation':'exploratory selected-node utility, no prospectively defined clinical action',
 'outcome':'recorded node status assigned under original level-based pathology matching scheme',
 'patient_level_outcome':'at least one malignant selected node; max node score is not a calibrated patient probability',
 'class_rebalancing':'none','univariate_feature_screening':'none','external_validation':'none',
 'source_sha256':sha256(SOURCE.read_bytes()).hexdigest(),
}

def json_save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def csv_save(df,p): df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')
def make_pipeline(estimator):
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler()),('model',estimator)])
def grouped_splits(frame,folds,seed):
    cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    split=list(cv.split(frame,frame.outcome,frame.group_id))
    for tr,te in split:
        assert not set(frame.iloc[tr].group_id)&set(frame.iloc[te].group_id)
        assert frame.iloc[tr].outcome.nunique()==2 and frame.iloc[te].outcome.nunique()==2
    return split
def get_threshold(y,p):
    fpr,tpr,t=roc_curve(y,p)
    ok=np.isfinite(t)
    best=np.max((tpr-fpr)[ok])
    return float(np.max(t[ok & np.isclose(tpr-fpr,best,atol=1e-12,rtol=0)]))

def prepare():
    frozen=ROOT/'analysis_plan_frozen.json'
    if frozen.exists():
        existing=json.loads(frozen.read_text(encoding='utf-8'))
        assert existing['plan']==METHODS,'The frozen analysis plan changed.'
    else: json_save(frozen,{'frozen_at':datetime.now().isoformat(),'plan':METHODS})
    data=pd.read_csv(SOURCE,dtype={'node_id':str,'patient_group_source':str})
    refs=pd.read_csv(REFERENCE,dtype={'node_id':str,'patient_group_source':str})
    data=data.merge(refs[['node_id','ce_ct_artery','ce_ct_muscle']],on='node_id',how='left',validate='one_to_one')
    data.loc[:,'group_id']=data.patient_group_source.astype(str)
    assert len(data)==254 and data.node_id.nunique()==254
    assert data.outcome.sum()==88 and data.group_id.nunique()==131
    data.loc[:,'outer_fold']=0
    for f,(_,te) in enumerate(grouped_splits(data,5,SEED),1):data.loc[te,'outer_fold']=f
    assert data.groupby('group_id').outer_fold.nunique().max()==1
    csv_save(data,AN/'cohort_all.csv')
    primary=data[data.short_axis_mm>3].copy().reset_index(drop=True)
    assert len(primary)==248 and primary.group_id.nunique()==130 and primary.outcome.sum()==84
    csv_save(primary,AN/'cohort_primary.csv')
    csv_save(data[['node_id','group_id','outcome','outer_fold']],AN/'outer_fold_assignment.csv')
    fold_counts=[]
    for f in range(1,6):
        for partition,mask in [('development',primary.outer_fold.ne(f)),('evaluation',primary.outer_fold.eq(f))]:
            q=primary[mask]
            fold_counts.append({'fold':f,'partition':partition,'nodes':len(q),'groups':q.group_id.nunique(),'malignant_nodes':int(q.outcome.sum()),'benign_nodes':int((1-q.outcome).sum()),'groups_with_malignant_selected_node':int(q.groupby('group_id').outcome.max().sum())})
    csv_save(pd.DataFrame(fold_counts),AN/'fold_counts.csv')
    json_save(AN/'software_versions.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'joblib':joblib.__version__,'BLAS_threads':1,'grid_search_jobs':1})
    return data,primary

def nested_main(frame,feature_list,scenario,explain=False):
    target=AN if scenario=='primary' else AN/'sensitivity'/scenario
    target.mkdir(exist_ok=True,parents=True)
    if scenario!='primary':csv_save(frame,target/'cohort.csv')
    prediction_path=target/('primary_predictions.csv' if scenario=='primary' else 'predictions.csv')
    pred_rows=[];fit_rows=[];tuning_rows=[];shap_rows=[];example_info=[];overlap_checks=[]
    specs={'NodeRADS':['node_rads'],'Imaging':feature_list,'Combined':feature_list+['node_rads']}
    for f in range(1,6):
        tr=frame[frame.outer_fold.ne(f)].reset_index(drop=True)
        te=frame[frame.outer_fold.eq(f)].reset_index(drop=True)
        assert not set(tr.group_id)&set(te.group_id)
        inner=grouped_splits(tr,3,SEED+f)
        overlap_checks.append({'scenario':scenario,'outer_fold':f,'outer_group_overlap':0,'inner_group_overlap':0,'development_nodes':len(tr),'evaluation_nodes':len(te)})
        for name,features in specs.items():
            start=time.perf_counter()
            pipe=make_pipeline(LogisticRegression(max_iter=5000,solver='lbfgs',random_state=SEED))
            grid=GridSearchCV(pipe,{'model__C':CS},scoring='neg_log_loss',cv=inner,n_jobs=1,refit=True,error_score='raise')
            grid.fit(tr[features],tr.outcome)
            inner_p=cross_val_predict(clone(grid.best_estimator_),tr[features],tr.outcome,cv=inner,method='predict_proba',n_jobs=1)[:,1]
            threshold=get_threshold(tr.outcome.to_numpy(),inner_p)
            fit_seconds=time.perf_counter()-start
            start=time.perf_counter()
            p=grid.predict_proba(te[features])[:,1]
            inference_seconds=time.perf_counter()-start
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            for r,pp in zip(te.itertuples(),p):
                pred_rows.append({'node_id':r.node_id,'group_id':r.group_id,'fold':f,'y_true':int(r.outcome),'model':name,'probability':float(pp),'prediction':int(pp>=threshold),'threshold':threshold})
            fit_rows.append({'scenario':scenario,'model':name,'fold':f,'features':len(features),'C':grid.best_params_['model__C'],'threshold':threshold,'best_inner_log_loss':-grid.best_score_,'fit_and_threshold_seconds':fit_seconds,'inference_seconds':inference_seconds,'evaluation_nodes':len(te),'evaluation_auc':roc_auc_score(te.outcome,p),'evaluation_average_precision':average_precision_score(te.outcome,p)})
            for pars,score,std in zip(grid.cv_results_['params'],grid.cv_results_['mean_test_score'],grid.cv_results_['std_test_score']):
                tuning_rows.append({'scenario':scenario,'model':name,'outer_fold':f,'C':pars['model__C'],'mean_inner_log_loss':-score,'sd_inner_log_loss':std})
            if explain:
                joblib.dump(grid.best_estimator_,AN/'models'/f'{name}_outer_fold_{f}.joblib')
            if explain and name=='Combined':
                import shap
                fitted=grid.best_estimator_
                ztrain=fitted[:-1].transform(tr[features]); ztest=fitted[:-1].transform(te[features])
                masker=shap.maskers.Independent(ztrain,max_samples=len(ztrain))
                explainer=shap.LinearExplainer(fitted.named_steps['model'],masker)
                explanation=explainer(ztest)
                reconstructed=explanation.base_values+explanation.values.sum(axis=1)
                decision=fitted.decision_function(te[features])
                assert np.max(np.abs(reconstructed-decision))<1e-9
                assert np.max(np.abs(expit(reconstructed)-p))<1e-10
                for ix,row in enumerate(te.itertuples()):
                    example_info.append({'node_id':row.node_id,'group_id':row.group_id,'fold':f,'base_value':float(explanation.base_values[ix]),'model_output':float(decision[ix]),'predicted_probability':float(p[ix]),'y_true':int(row.outcome)})
                    for j,feature in enumerate(features):
                        shap_rows.append({'node_id':row.node_id,'group_id':row.group_id,'fold':f,'model':name,'feature':feature,'feature_value':float(te.iloc[ix][feature]) if pd.notna(te.iloc[ix][feature]) else np.nan,'imputed_feature_value':float(fitted.named_steps['imputer'].transform(te[features])[ix,j]),'shap_value':float(explanation.values[ix,j]),'base_value':float(explanation.base_values[ix]),'model_output':float(decision[ix])})
            print(f'{scenario}: fold {f}/5 {name} completed',flush=True)
    predictions=pd.DataFrame(pred_rows)
    assert predictions.groupby(['node_id','model']).size().eq(1).all()
    assert len(predictions)==len(frame)*3
    csv_save(predictions,prediction_path)
    csv_save(pd.DataFrame(fit_rows),target/'fitting_details.csv')
    csv_save(pd.DataFrame(tuning_rows),target/'inner_tuning_results.csv')
    csv_save(pd.DataFrame(overlap_checks),target/'group_overlap_checks.csv')
    if explain:
        csv_save(pd.DataFrame(shap_rows),AN/'shap_values_long.csv')
        examples=[]
        for y in [0,1]:
            candidates=[r for r in example_info if r['y_true']==y]
            examples.append(min(candidates,key=lambda r:(abs(r['predicted_probability']-.5),r['node_id'])))
        json_save(AN/'shap_examples.json',{'output_scale':'log_odds','examples':examples,'selection_rule':METHODS['SHAP_examples'],'background':'all corresponding outer-training nodes','additivity_verified':True})
        coefficients=[]
        for name,features in specs.items():
            inner=grouped_splits(frame,3,SEED+100)
            grid=GridSearchCV(make_pipeline(LogisticRegression(max_iter=5000,solver='lbfgs',random_state=SEED)),{'model__C':CS},scoring='neg_log_loss',cv=inner,n_jobs=1,refit=True,error_score='raise')
            grid.fit(frame[features],frame.outcome)
            fitted=grid.best_estimator_; model=fitted.named_steps['model']; scaler=fitted.named_steps['scaler']; imputer=fitted.named_steps['imputer']
            beta=model.coef_[0]/scaler.scale_
            intercept=float(model.intercept_[0]-np.sum(model.coef_[0]*scaler.mean_/scaler.scale_))
            coefficients.append({'model':name,'feature':'Intercept','coefficient_original_units':intercept,'standardized_coefficient':float(model.intercept_[0]),'imputation_median':np.nan,'training_mean':np.nan,'training_sd':np.nan,'C':grid.best_params_['model__C']})
            for j,c in enumerate(features):
                coefficients.append({'model':name,'feature':c,'coefficient_original_units':float(beta[j]),'standardized_coefficient':float(model.coef_[0,j]),'imputation_median':float(imputer.statistics_[j]),'training_mean':float(scaler.mean_[j]),'training_sd':float(scaler.scale_[j]),'C':grid.best_params_['model__C']})
            x=imputer.transform(frame[features]); re_p=expit(intercept+x@beta)
            assert np.max(abs(re_p-fitted.predict_proba(frame[features])[:,1]))<1e-10
            joblib.dump(fitted,AN/'models'/f'{name}_full_development_model.joblib')
        csv_save(pd.DataFrame(coefficients),AN/'final_model_coefficients.csv')
    return predictions

def exploratory(frame):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import xgboost,lightgbm
    definitions={
      'Logistic regression':(LogisticRegression(max_iter=5000,random_state=SEED),{'model__C':CS}),
      'Naive Bayes':(GaussianNB(),{'model__var_smoothing':[1e-9,1e-6,1e-3]}),
      'SVM':(SVC(probability=False,random_state=SEED),{'model__C':[.1,1,10],'model__gamma':['scale',.1]}),
      'KNN':(KNeighborsClassifier(),{'model__n_neighbors':[3,5,9,15],'model__weights':['uniform','distance']}),
      'Random forest':(RandomForestClassifier(n_estimators=200,n_jobs=1,random_state=SEED),{'model__max_depth':[3,None],'model__min_samples_leaf':[2,5]}),
      'Extra Trees':(ExtraTreesClassifier(n_estimators=200,n_jobs=1,random_state=SEED),{'model__max_depth':[3,None],'model__min_samples_leaf':[2,5]}),
      'XGBoost':(XGBClassifier(n_estimators=150,n_jobs=1,random_state=SEED,eval_metric='logloss',tree_method='hist'),{'model__max_depth':[2,3],'model__learning_rate':[.03,.1]}),
      'LightGBM':(LGBMClassifier(n_estimators=150,n_jobs=1,random_state=SEED,verbosity=-1,min_child_samples=10),{'model__num_leaves':[5,10],'model__learning_rate':[.03,.1]}),
      'AdaBoost':(AdaBoostClassifier(random_state=SEED),{'model__n_estimators':[50,100],'model__learning_rate':[.1,.5]}),
      'Gradient boosting':(GradientBoostingClassifier(n_estimators=100,random_state=SEED),{'model__max_depth':[1,2],'model__learning_rate':[.03,.1]}),
      'MLP':(MLPClassifier(solver='lbfgs',max_iter=3000,max_fun=30000,random_state=SEED,early_stopping=False),{'model__hidden_layer_sizes':[(8,),(16,)],'model__alpha':[.1,1.0]}),
    }
    metrics=[];preds=[];specs=[]
    for algorithm,(estimator,params) in definitions.items():
      for f in range(1,6):
        tr=frame[frame.outer_fold.ne(f)].reset_index(drop=True);te=frame[frame.outer_fold.eq(f)].reset_index(drop=True)
        inner=grouped_splits(tr,3,SEED+f)
        start=time.perf_counter()
        grid=GridSearchCV(make_pipeline(clone(estimator)),params,scoring='roc_auc',cv=inner,n_jobs=1,error_score='raise')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            grid.fit(tr[FEATURES],tr.outcome)
            start_predict=time.perf_counter()
            score=grid.decision_function(te[FEATURES]) if algorithm=='SVM' else grid.predict_proba(te[FEATURES])[:,1]
            predict_seconds=time.perf_counter()-start_predict
        metrics.append({'algorithm':algorithm,'fold':f,'auc':roc_auc_score(te.outcome,score),'average_precision':average_precision_score(te.outcome,score),'best_inner_auc':grid.best_score_,'training_seconds':time.perf_counter()-start,'inference_seconds':predict_seconds,'evaluation_nodes':len(te),'warnings':'; '.join(sorted(set(type(w.message).__name__ for w in captured)))})
        specs.append({'algorithm':algorithm,'fold':f,'parameters':grid.best_params_})
        for row,sc in zip(te.itertuples(),score):preds.append({'algorithm':algorithm,'fold':f,'node_id':row.node_id,'group_id':row.group_id,'y_true':row.outcome,'score':float(sc),'score_type':'decision_function' if algorithm=='SVM' else 'probability'})
        print(f'Exploratory {algorithm}: fold {f}/5 completed',flush=True)
      csv_save(pd.DataFrame(metrics),AN/'exploratory_fold_metrics.csv')
      csv_save(pd.DataFrame(preds),AN/'exploratory_predictions.csv')
      json_save(AN/'exploratory_model_settings.json',{'grids':{n:p for n,(e,p) in definitions.items()},'selected_settings':specs,'versions':{'xgboost':xgboost.__version__,'lightgbm':lightgbm.__version__}})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['primary','sensitivity','exploratory','all'],default='all');args=parser.parse_args()
    all_data,primary=prepare()
    with threadpool_limits(limits=1),joblib.parallel_backend('threading',n_jobs=1):
      if args.stage in ['primary','all']:nested_main(primary,FEATURES,'primary',True)
      if args.stage in ['sensitivity','all']:
        scenarios=[('all_254',all_data,FEATURES),('legacy_249',all_data[all_data.included_in_previous_249].copy(),FEATURES),
          ('reference_augmented',primary,FEATURES+['ce_ct_artery','ce_ct_muscle']),
          ('normalized_CT',primary,['ct_ratio_artery','ct_ratio_muscle','short_axis_mm','long_axis_mm']),
          ('exclude_group_76',primary[primary.group_id.ne('76')].copy(),FEATURES)]
        for name,frame,features in scenarios:nested_main(frame.reset_index(drop=True),features,name)
      if args.stage in ['exploratory','all']:exploratory(primary)

if __name__=='__main__':main()

