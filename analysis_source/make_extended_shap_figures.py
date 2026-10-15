"""Restore imaging-only and force-plot explanations using existing outer models.

No model fitting, tuning, or primary prediction changes are performed.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[v]='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from threadpoolctl import threadpool_limits
from make_figures import PRETTY, setup_style, Plotter

ROOT=Path(__file__).resolve().parents[1]
AN=ROOT/'analysis'
FIG=ROOT/'figures_tables'/'figures'
SOURCE=ROOT/'figures_tables'/'source_data'
FEATURES=['nc_ct','ce_ct','nc_iodine','ce_iodine','nc_zeff','ce_zeff','nc_ed','ce_ed',
          'ct_70kev','ct_100kev','casupp_30','casupp_70','short_axis_mm','long_axis_mm']
RED='#B2182B'
BLUE='#2166AC'

def csv_save(d,p):
    d.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')

def save_figure(fig,stem,width,height):
    for ext in ['pdf','svg','png','tif']:
        fig.savefig(FIG/(stem+'.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    return {'stem':stem,'width_mm':width,'height_mm':height,'raster_dpi':300,'font':'Arial'}

def imaging_explanations(cohort,predictions):
    rows=[]
    checks=[]
    with threadpool_limits(limits=1):
      for fold in range(1,6):
        tr=cohort[cohort.outer_fold.ne(fold)].reset_index(drop=True)
        te=cohort[cohort.outer_fold.eq(fold)].reset_index(drop=True)
        assert not set(tr.group_id)&set(te.group_id)
        fitted=joblib.load(AN/'models'/f'Imaging_outer_fold_{fold}.joblib')
        assert list(fitted.feature_names_in_)==FEATURES
        imputer=fitted.named_steps['imputer']
        scaler=fitted.named_steps['scaler']
        np.testing.assert_allclose(imputer.statistics_,np.nanmedian(tr[FEATURES].to_numpy(),axis=0),rtol=1e-12,atol=1e-12)
        xtrain=imputer.transform(tr[FEATURES])
        np.testing.assert_allclose(scaler.mean_,xtrain.mean(axis=0),rtol=1e-12,atol=1e-12)
        sd=xtrain.std(axis=0,ddof=0);sd[sd==0]=1
        np.testing.assert_allclose(scaler.scale_,sd,rtol=1e-12,atol=1e-12)
        p=fitted.predict_proba(te[FEATURES])[:,1]
        saved=predictions[predictions.model.eq('Imaging')&predictions.fold.eq(fold)].set_index('node_id').loc[te.node_id]
        perror=float(np.max(abs(p-saved.probability.to_numpy())))
        assert perror<1e-10
        ztrain=fitted[:-1].transform(tr[FEATURES]);ztest=fitted[:-1].transform(te[FEATURES])
        masker=shap.maskers.Independent(ztrain,max_samples=len(ztrain))
        explanation=shap.LinearExplainer(fitted.named_steps['model'],masker)(ztest)
        decision=fitted.decision_function(te[FEATURES])
        sum_value=explanation.base_values+explanation.values.sum(axis=1)
        add_error=float(np.max(abs(sum_value-decision)))
        shap_perror=float(np.max(abs(expit(sum_value)-p)))
        assert add_error<1e-9 and shap_perror<1e-10
        xtest=imputer.transform(te[FEATURES])
        for i,node in enumerate(te.itertuples()):
          for j,f in enumerate(FEATURES):
            rows.append({'node_id':node.node_id,'group_id':node.group_id,'fold':fold,'model':'Imaging',
                         'feature':f,'feature_value':float(te.iloc[i][f]) if pd.notna(te.iloc[i][f]) else np.nan,
                         'imputed_feature_value':float(xtest[i,j]),'shap_value':float(explanation.values[i,j]),
                         'base_value':float(explanation.base_values[i]),'model_output':float(decision[i]),
                         'probability':float(p[i]),'y_true':int(node.outcome)})
        checks.append({'fold':fold,'development_nodes':len(tr),'evaluation_nodes':len(te),
                       'source_group_overlap':0,'all_development_nodes_in_background':True,
                       'imputer_scaler_match_development':True,'max_saved_probability_error':perror,
                       'max_logit_additivity_error':add_error,'max_probability_additivity_error':shap_perror,
                       'model_refitted':False})
        print('Existing Imaging model fold',fold,'explained',flush=True)
    data=pd.DataFrame(rows)
    assert len(data)==248*14 and data.groupby(['node_id','feature']).size().eq(1).all()
    csv_save(data,AN/'imaging_heldout_shap_values.csv')
    csv_save(data,SOURCE/'SupplementaryFigure14_imaging_heldout_SHAP_values.csv')
    csv_save(pd.DataFrame(checks),AN/'imaging_heldout_shap_checks.csv')
    importance=(data.groupby('feature').shap_value.apply(lambda s:float(abs(s).mean()))
                .sort_values(ascending=False).rename('mean_absolute_shap_log_odds').reset_index())
    importance['rank']=np.arange(1,len(importance)+1)
    importance['nodes']=248
    csv_save(importance,SOURCE/'SupplementaryFigure14_imaging_SHAP_importance.csv')
    order=importance.feature.tolist()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(180/25.4,116/25.4),gridspec_kw={'width_ratios':[.85,1.15]})
    fig.subplots_adjust(left=.31,right=.86,bottom=.13,top=.87,wspace=.16)
    y=np.arange(14)[::-1]
    ax.barh(y,importance.mean_absolute_shap_log_odds,height=.68,color='#D55E00')
    ax.set_yticks(y,[PRETTY.get(f,f) for f in order]);ax.tick_params(axis='y',length=0)
    ax.set_ylim(-.7,13.7);ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('A  Global contribution',loc='left',pad=9)
    cmap=LinearSegmentedColormap.from_list('feature_value',[BLUE,'#F7F7F7',RED])
    scatter=None
    for yi,f in zip(y,order):
        q=data[data.feature.eq(f)].sort_values('node_id')
        sv=q.shap_value.to_numpy()
        color=q.imputed_feature_value.rank(method='average',pct=True).to_numpy()
        scatter=bx.scatter(sv,yi+Plotter.beeswarm_offsets(sv),c=color,cmap=cmap,vmin=0,vmax=1,
                           s=7,alpha=.80,linewidths=0)
    bx.axvline(0,color='#777777',lw=.6,zorder=0)
    bx.set_yticks(y);bx.set_yticklabels([]);bx.tick_params(axis='y',length=0)
    bx.set_ylim(ax.get_ylim());bx.spines.left.set_visible(False)
    bx.set_xlabel('SHAP value (log-odds)')
    bx.set_title('B  Direction and input value',loc='left',pad=9)
    cax=fig.add_axes([.90,.29,.010,.43])
    cb=fig.colorbar(scatter,cax=cax,ticks=[0,1]);cb.ax.set_yticklabels(['Low','High'])
    cb.set_label('Within-feature input rank',fontsize=7.5,labelpad=2)
    cb.outline.set_visible(False)
    fig.text(.025,.955,'Imaging-only ridge model: 14 inputs; 248 held-out nodes',fontsize=8.5)
    fig.text(.025,.033,'Each outer model uses its own development background. Contributions describe model calculations.',fontsize=7.5)
    output=save_figure(fig,'SupplementaryFigure14_Imaging_SHAP',180,116)
    caption=('Supplementary Figure 14. Held-out explanation of the imaging-only ridge model. '
             'A, mean absolute SHAP contributions for all 14 direct quantitative node measurements. '
             'B, one point per feature and held-out node, ordered as in A, with colors representing within-feature '
             'ranks of the actual inputs after training-specific imputation. The 248 nodes belong to 130 source-recorded groups. '
             'The existing five outer-fold Imaging pipelines were used without fitting or tuning. All corresponding '
             'outer-development nodes formed each linear interventional background after training-specific preprocessing. '
             'SHAP is expressed in log-odds units, and additivity and saved held-out probabilities were verified for every node. '
             'Correlated measurements can share attribution, and these contributions are not biological effects.\n')
    (FIG/'SupplementaryFigure14_caption.txt').write_text(caption,encoding='utf-8')
    return checks,importance,output

def display_value(v):
    return f'{v:.3g}'

def force_examples():
    combined=pd.read_csv(AN/'shap_values_long.csv',dtype={'node_id':str,'group_id':str})
    info=json.loads((AN/'shap_examples.json').read_text(encoding='utf-8'))
    examples=info['examples']
    assert len(examples)==2 and {q['y_true'] for q in examples}=={0,1}
    displayed=[];all_source=[];checks=[]
    fig=plt.figure(figsize=(180/25.4,165/25.4))
    axes=[fig.add_axes([.075,.67,.89,.235]),fig.add_axes([.075,.215,.89,.235])]
    for panel,(ax,example) in enumerate(zip(axes,examples)):
        q=combined[combined.node_id.eq(example['node_id'])].copy()
        assert len(q)==15 and q.feature.nunique()==15
        assert q.fold.nunique()==1 and int(q.fold.iloc[0])==example['fold']
        baseline=float(q.base_value.iloc[0]);endpoint=float(q.model_output.iloc[0])
        total=float(q.shap_value.sum());probability=float(example['predicted_probability'])
        add_error=abs(baseline+total-endpoint)
        p_error=abs(float(expit(endpoint))-probability)
        assert add_error<1e-9 and p_error<1e-10
        q=q.sort_values('shap_value',key=lambda x:abs(x),ascending=False).reset_index(drop=True)
        top=q.iloc[:6]
        other=q.iloc[6:]
        shown=[]
        for i,row in enumerate(top.itertuples(),1):
            shown.append({'display_id':i,'feature':row.feature,'feature_label':PRETTY.get(row.feature,row.feature),
                          'imputed_feature_value':float(row.imputed_feature_value),
                          'shap_value':float(row.shap_value),'included_features':row.feature})
        shown.append({'display_id':7,'feature':'Other','feature_label':'Other 9 inputs',
                      'imputed_feature_value':np.nan,'shap_value':float(other.shap_value.sum()),
                      'included_features':' | '.join(other.feature)})
        assert abs(sum(r['shap_value'] for r in shown)-total)<1e-12
        positive=[r for r in shown if r['shap_value']>=0]
        negative=[r for r in shown if r['shap_value']<0]
        pos_sum=sum(r['shap_value'] for r in positive)
        neg_sum=sum(r['shap_value'] for r in negative)
        left=endpoint-pos_sum;right=endpoint-neg_sum
        span=max(right-left,.1)
        padding=span*.11
        ax.set_xlim(left-padding,right+padding)
        ax.set_ylim(0,1.0)
        y0=.38;y1=.57
        current=left
        for row in positive:
            start=current;end=current+row['shap_value'];width=end-start
            tip=min(width*.20,span*.024)
            points=[(start,y0),(end-tip,y0),(end,(y0+y1)/2),(end-tip,y1),(start,y1)]
            ax.add_patch(Polygon(points,facecolor=RED,edgecolor='white',linewidth=.4))
            row['force_start']=start;row['force_end']=end
            row['direction']='increases_output'
            row['segment_midpoint']=(start+end)/2
            current=end
        assert abs(current-endpoint)<1e-12
        current=right
        for row in negative:
            start=current;end=current+row['shap_value'];width=start-end
            tip=min(width*.20,span*.024)
            points=[(start,y0),(end+tip,y0),(end,(y0+y1)/2),(end+tip,y1),(start,y1)]
            ax.add_patch(Polygon(points,facecolor=BLUE,edgecolor='white',linewidth=.4))
            row['force_start']=start;row['force_end']=end
            row['direction']='decreases_output'
            row['segment_midpoint']=(start+end)/2
            current=end
        assert abs(current-endpoint)<1e-12
        for row in shown:
            width=abs(row['force_end']-row['force_start'])
            mid=row['segment_midpoint']
            if width/span>.055:
                ax.text(mid,(y0+y1)/2,str(row['display_id']),ha='center',va='center',fontsize=7.5,color='black',
                        bbox={'boxstyle':'circle,pad=.10','fc':'white','ec':'none'})
            else:
                label_y=.66 + .045*(row['display_id']%2)
                ax.annotate(str(row['display_id']),(mid,y1),xytext=(mid,label_y),ha='center',va='bottom',
                            fontsize=7.0,arrowprops={'arrowstyle':'-','lw':.45,'color':'#555555'})
            displayed.append({'node_id':example['node_id'],'fold':example['fold'],'y_true':example['y_true'],
                              'base_value':baseline,'model_output':endpoint,'probability':probability,**row})
        ax.vlines(baseline,.26,.35,color='#777777',linestyle='--',linewidth=.75)
        ax.vlines(endpoint,.35,.61,color='black',linewidth=.8)
        ax.text(baseline,.285,f'Base {baseline:.3f}',ha='center',va='bottom',fontsize=7.5,
                bbox={'facecolor':'white','edgecolor':'none','pad':.2})
        ax.text(endpoint,.89,f'Output {endpoint:.3f}',ha='center',va='top',fontsize=7.5)
        title=('A' if panel==0 else 'B')+'  Recorded '+('benign' if example['y_true']==0 else 'malignant')
        ax.set_title(f"{title}; held-out malignant probability {probability:.3f}",loc='left',fontsize=8.5,pad=7)
        ax.set_yticks([])
        ax.spines.left.set_visible(False)
        ax.spines.bottom.set_position(('axes',.25))
        ax.tick_params(axis='x',labelsize=7.5,pad=2)
        ax.text(.5,.025,'Log-odds of recorded malignancy',ha='center',va='bottom',
                fontsize=7.5,transform=ax.transAxes)
        # Label key below each force panel. Every displayed block corresponds to one entry.
        for i,row in enumerate(shown):
            col=i%3;line=i//3
            xf=.075+col*.303
            label=row['feature_label']
            if row['feature']!='Other':
                label+=f" = {display_value(row['imputed_feature_value'])}"
            yf=(.625 if panel==0 else .17)-line*.045
            fig.text(xf,yf,f"{row['display_id']}  {label}\nSHAP {row['shap_value']:+.3f}",
                     ha='left',va='top',fontsize=7.5,color='black',linespacing=1.1)
        for row in q.itertuples():
            all_source.append({'node_id':example['node_id'],'fold':example['fold'],'y_true':example['y_true'],
                               'feature':row.feature,'feature_value':row.feature_value,
                               'imputed_feature_value':row.imputed_feature_value,'shap_value':row.shap_value,
                               'base_value':baseline,'model_output':endpoint,'probability':probability,
                               'display_group':int(top.index[top.feature.eq(row.feature)][0])+1
                               if row.feature in set(top.feature) else 7})
        checks.append({'node_id':example['node_id'],'fold':example['fold'],'features':15,
                       'shown_individually':6,'remaining_aggregated':9,'base_value':baseline,'model_output':endpoint,
                       'probability':probability,'max_logit_error':add_error,'max_probability_error':p_error,
                       'sum_display_contributions':sum(r['shap_value'] for r in shown),'sum_all_contributions':total,
                       'selection_rule':info['selection_rule']})
    fig.text(.075,.025,'Red increases and blue decreases log-odds. Examples were selected near probability 0.5, not for successful classification.',fontsize=7.5)
    output=save_figure(fig,'SupplementaryFigure15_Combined_SHAP_force',180,165)
    csv_save(pd.DataFrame(displayed),SOURCE/'SupplementaryFigure15_force_display_groups.csv')
    csv_save(pd.DataFrame(all_source),SOURCE/'SupplementaryFigure15_all_example_SHAP_values.csv')
    caption=('Supplementary Figure 15. Combined-model force diagrams for the two prespecified borderline held-out examples. '
             'A, a recorded benign node with malignant probability 0.500. B, a recorded malignant node with probability 0.492. '
             'The examples were selected by closeness to probability 0.5 within each recorded outcome class, not by correct classification. '
             'Red blocks increase and blue blocks decrease the corresponding outer model output on the log-odds scale. '
             'The six largest absolute contributions are displayed individually, and the signed sum of the remaining nine inputs '
             'is shown as Other. All 15 contributions are retained in the source data and sum with the baseline to the stated output. '
             'The inverse logit reproduces each saved held-out probability. Colors denote contribution direction, and these calculations '
             'do not indicate biological effects or an established treatment threshold.\n')
    (FIG/'SupplementaryFigure15_caption.txt').write_text(caption,encoding='utf-8')
    return checks,output

def main():
    setup_style()
    plt.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5})
    cohort=pd.read_csv(AN/'cohort_primary.csv',dtype={'node_id':str,'group_id':str})
    predictions=pd.read_csv(AN/'primary_predictions.csv',dtype={'node_id':str,'group_id':str})
    assert len(cohort)==248 and cohort.group_id.nunique()==130
    imaging_checks,importance,s14=imaging_explanations(cohort,predictions)
    force_checks,s15=force_examples()
    qa={'new_model_fits':0,'model_tuning_performed':False,'primary_predictions_modified':False,
        'imaging_checks':imaging_checks,'force_checks':force_checks,'figures':[s14,s15],
        'imaging_nodes':248,'recorded_groups':130,'imaging_features':14,'combined_features':15,
        'shap_version':shap.__version__,'backend':matplotlib.get_backend()}
    (AN/'extended_shap_figure_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(importance.head(5).to_string(index=False))
    print('Extended SHAP figures saved. New model fits:',qa['new_model_fits'])

if __name__=='__main__':
    main()
