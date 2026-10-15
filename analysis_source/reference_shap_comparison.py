"""Reproduce reference-augmented outer fits for paired held-out SHAP comparison.

Uses saved selected C values without repeating search or changing the primary analysis.
"""
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / 'analysis'
TARGET = AN / 'sensitivity' / 'reference_augmented'
FIG = ROOT / 'figures_tables' / 'figures'
SOURCE = ROOT / 'figures_tables' / 'source_data'
FEATURES = ['nc_ct','ce_ct','nc_iodine','ce_iodine','nc_zeff','ce_zeff',
            'nc_ed','ce_ed','ct_70kev','ct_100kev','casupp_30','casupp_70',
            'short_axis_mm','long_axis_mm','ce_ct_artery','ce_ct_muscle','node_rads']
LABELS = {'nc_ct':'Noncontrast nodal CT','ce_ct':'Enhanced nodal CT',
          'nc_iodine':'Noncontrast nodal iodine','ce_iodine':'Enhanced nodal iodine',
          'nc_zeff':'Noncontrast effective atomic number','ce_zeff':'Enhanced effective atomic number',
          'nc_ed':'Noncontrast electron density','ce_ed':'Enhanced electron density',
          'ct_70kev':'70-keV nodal attenuation','ct_100kev':'100-keV nodal attenuation',
          'casupp_30':'CaSupp 30 nodal attenuation','casupp_70':'CaSupp 70 nodal attenuation',
          'short_axis_mm':'Short-axis diameter','long_axis_mm':'Long-axis diameter',
          'ce_ct_artery':'Enhanced arterial reference CT',
          'ce_ct_muscle':'Enhanced muscle reference CT','node_rads':'Node-RADS'}

def csv_save(frame, path):
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.12g')

def main():
    cohort = pd.read_csv(TARGET/'cohort.csv',dtype={'node_id':str,'group_id':str})
    predictions = pd.read_csv(TARGET/'predictions.csv',dtype={'node_id':str,'group_id':str})
    settings = pd.read_csv(TARGET/'fitting_details.csv')
    primary = pd.read_csv(AN/'shap_values_long.csv',dtype={'node_id':str,'group_id':str})
    assert len(cohort)==248 and cohort.group_id.nunique()==130
    assert set(primary.node_id)==set(cohort.node_id)
    rows, checks = [], []
    with threadpool_limits(limits=1):
      for fold in range(1,6):
        tr = cohort[cohort.outer_fold.ne(fold)].reset_index(drop=True)
        te = cohort[cohort.outer_fold.eq(fold)].reset_index(drop=True)
        assert not set(tr.group_id) & set(te.group_id)
        c = float(settings.loc[settings.model.eq('Combined') & settings.fold.eq(fold),'C'].iloc[0])
        fitted = Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                           ('scaler',StandardScaler()),
                           ('model',LogisticRegression(C=c,max_iter=5000,solver='lbfgs',random_state=20260926))])
        fitted.fit(tr[FEATURES],tr.outcome)
        imputer = fitted.named_steps['imputer']
        scaler = fitted.named_steps['scaler']
        medians = np.nanmedian(tr[FEATURES].to_numpy(),axis=0)
        assert np.max(np.abs(imputer.statistics_-medians))<1e-12
        xtrain = imputer.transform(tr[FEATURES])
        np.testing.assert_allclose(scaler.mean_,xtrain.mean(axis=0),rtol=1e-12,atol=1e-12)
        expected_sd = xtrain.std(axis=0,ddof=0)
        expected_sd[expected_sd==0] = 1
        np.testing.assert_allclose(scaler.scale_,expected_sd,rtol=1e-12,atol=1e-12)
        p = fitted.predict_proba(te[FEATURES])[:,1]
        saved = predictions[predictions.model.eq('Combined') & predictions.fold.eq(fold)].set_index('node_id')
        saved_p = saved.loc[te.node_id,'probability'].to_numpy()
        probability_error = float(np.max(abs(saved_p-p)))
        assert probability_error<1e-10,(fold,probability_error)
        ztrain = fitted[:-1].transform(tr[FEATURES])
        ztest = fitted[:-1].transform(te[FEATURES])
        explainer = shap.LinearExplainer(fitted.named_steps['model'],
                                        shap.maskers.Independent(ztrain,max_samples=len(ztrain)))
        explanation = explainer(ztest)
        logit = fitted.decision_function(te[FEATURES])
        reconstructed = explanation.base_values + explanation.values.sum(axis=1)
        logit_error = float(np.max(abs(reconstructed-logit)))
        shap_probability_error = float(np.max(abs(expit(reconstructed)-p)))
        assert logit_error<1e-9 and shap_probability_error<1e-10
        imputed = imputer.transform(te[FEATURES])
        for i,node in enumerate(te.itertuples()):
          for j,feature in enumerate(FEATURES):
            rows.append({'node_id':node.node_id,'group_id':node.group_id,'fold':fold,
                         'model':'Reference_augmented_combined','feature':feature,
                         'feature_value':float(te.iloc[i][feature]) if pd.notna(te.iloc[i][feature]) else np.nan,
                         'imputed_feature_value':float(imputed[i,j]),'shap_value':float(explanation.values[i,j]),
                         'base_value':float(explanation.base_values[i]),'model_output':float(logit[i]),
                         'probability':float(p[i])})
        checks.append({'fold':fold,'C':c,'development_nodes':len(tr),'evaluation_nodes':len(te),
                       'group_overlap':0,'max_probability_reproduction_error':probability_error,
                       'max_shap_logit_error':logit_error,'max_shap_probability_error':shap_probability_error,
                       'background':'all_outer_development_nodes','imputer_scaler_training_only':True})
        print('Reference SHAP fold',fold,'complete',flush=True)
    reference = pd.DataFrame(rows)
    assert len(reference)==248*17 and reference.groupby(['node_id','feature']).size().eq(1).all()
    csv_save(reference,AN/'reference_augmented_shap_values.csv')
    csv_save(pd.DataFrame(checks),AN/'reference_augmented_shap_checks.csv')
    summaries = []
    for label, frame in [('Primary_combined',primary),('Reference_augmented_combined',reference)]:
        s = frame.groupby('feature',sort=False).shap_value.apply(lambda x:float(np.mean(abs(x)))).sort_values(ascending=False)
        for rank,(feature,value) in enumerate(s.items(),1):
            summaries.append({'input_set':label,'feature':feature,'mean_absolute_shap_log_odds':value,
                              'rank':rank,'nodes':248,'output_scale':'log_odds'})
    summary = pd.DataFrame(summaries)
    csv_save(summary,AN/'reference_shap_importance_comparison.csv')
    wide = summary.pivot(index='feature',columns='input_set',values=['mean_absolute_shap_log_odds','rank'])
    wide.columns = ['_'.join(k) for k in wide.columns]
    wide = wide.reset_index().copy()
    wide.loc[:,'primary_included'] = wide.feature.isin(primary.feature.unique())
    csv_save(wide,SOURCE/'SupplementaryFigure2_reference_SHAP_comparison.csv')
    csv_save(reference,SOURCE/'SupplementaryFigure2_reference_heldout_SHAP_values.csv')
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'text.color':'black','axes.labelcolor':'black',
                         'xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
    order = summary[summary.input_set.eq('Reference_augmented_combined')].sort_values('rank').feature.tolist()
    primary_s = summary[summary.input_set.eq('Primary_combined')].set_index('feature').mean_absolute_shap_log_odds
    reference_s = summary[summary.input_set.eq('Reference_augmented_combined')].set_index('feature').mean_absolute_shap_log_odds
    y = np.arange(len(order))
    fig,ax = plt.subplots(figsize=(180/25.4,146/25.4))
    fig.subplots_adjust(left=.39,right=.98,bottom=.10,top=.88)
    ax.barh(y+.18,[reference_s[k] for k in order],height=.32,color='#D55E00',label='With reference attenuation')
    present = np.array([k in primary_s for k in order])
    ax.barh(y[present]-.18,[primary_s[k] for k in np.array(order)[present]],height=.32,
            color='#009E73',label='Primary, without reference attenuation')
    for i,k in enumerate(order):
        if k not in primary_s:
            ax.text(.018,i-.18,'excluded',va='center',ha='left',fontsize=7,color='black')
    ax.set_yticks(y,[LABELS[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel('Mean absolute held-out SHAP value (log-odds)')
    ax.set_xlim(0,max(reference_s.max(),primary_s.max())*1.09)
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='lower left',bbox_to_anchor=(-.65,1.015),frameon=False,fontsize=8,ncol=1)
    fig.suptitle('Reference attenuation and held-out feature contributions',x=.02,y=.985,ha='left',fontsize=9)
    stem = FIG/'SupplementaryFigure2_Reference_SHAP_comparison'
    for extension in ['pdf','svg','png','tif']:
        fig.savefig(stem.with_suffix('.'+extension),dpi=300,facecolor='white')
    plt.close(fig)
    caption = ('Supplementary Figure 2. Held-out SHAP contributions with and without raw reference attenuation. '
               'Both combined models were evaluated on the same 248 selected nodes in 130 source-recorded groups, '
               'using the retained outer folds and their development-selected regularization settings. '
               'The reference-augmented fits were reconstructed without retuning, and their predictions were verified '
               'against the saved sensitivity predictions. Bars show mean absolute contributions in log-odds units, '
               'using all corresponding outer-development nodes as the background. Raw arterial and muscle attenuation '
               'are excluded from the primary input set rather than assigned zero importance. '
               'Ranks and magnitudes describe the fitted models and are not tests of biological or causal effects.\n')
    (FIG/'SupplementaryFigure2_caption.txt').write_text(caption,encoding='utf-8')
    report = {'nodes':248,'groups':130,'features_reference':17,'primary_features':15,
              'selected_C_from':'analysis/sensitivity/reference_augmented/fitting_details.csv',
              'max_probability_reproduction_error':max(q['max_probability_reproduction_error'] for q in checks),
              'max_shap_logit_error':max(q['max_shap_logit_error'] for q in checks),
              'rankings':summary.to_dict('records'),'figure_mm':[180,146],'raster_dpi':300,
              'no_retuning':True,'primary_models_modified':False,'shap_version':shap.__version__}
    (AN/'reference_shap_comparison_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(summary[summary.input_set.eq('Reference_augmented_combined')].head(8).to_string(index=False))
    print('Max saved probability error:',report['max_probability_reproduction_error'])

if __name__=='__main__':
    main()
