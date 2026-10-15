"""Additional source-backed figures requested on 2026-10-08.

Uses fixed measurements, coefficients and inner-search outputs. Does not select
features, refit models or change the primary validation results.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from make_figures import setup_style, PRETTY, COLORS, MM

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / 'analysis'
FIG = ROOT / 'figures_tables' / 'figures'
SRC = ROOT / 'figures_tables' / 'source_data'
FEATURES = ['nc_ct','ce_ct','nc_iodine','ce_iodine','nc_zeff','ce_zeff','nc_ed','ce_ed',
            'ct_70kev','ct_100kev','casupp_30','casupp_70','short_axis_mm','long_axis_mm','node_rads']
SHORT = ['NC CT','CE CT','NC IC','CE IC','NC Zeff','CE Zeff','NC ED','CE ED','70 keV','100 keV','CaSupp 30','CaSupp 70','Short axis','Long axis','Node-RADS']
UNITS = {x:'HU' for x in ['nc_ct','ce_ct','ct_70kev','ct_100kev','casupp_30','casupp_70']}
UNITS.update({x:'mg/mL' for x in ['nc_iodine','ce_iodine']})
UNITS.update({x:'%EDW' for x in ['nc_ed','ce_ed']})
UNITS.update({x:'mm' for x in ['short_axis_mm','long_axis_mm']})
CAPTIONS = []


def pretty(f):
    return PRETTY.get(f, f) + (f' ({UNITS[f]})' if f in UNITS else '')


def save(fig, stem, caption, source_files):
    fig.canvas.draw()
    paths = []
    for ext in ['pdf','svg','png','tif']:
        path = FIG / (stem + '.' + ext)
        kwargs = {'dpi':300, 'facecolor':'white'}
        if ext == 'tif':
            kwargs['pil_kwargs'] = {'compression':'tiff_lzw'}
        fig.savefig(path, **kwargs)
        paths.append({'name':path.name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    CAPTIONS.append({'stem':stem, 'caption':caption, 'sources':source_files, 'files':paths,
                     'width_mm':round(fig.get_size_inches()[0]*25.4,3), 'height_mm':round(fig.get_size_inches()[1]*25.4,3)})
    plt.close(fig)
    print(stem, flush=True)


def correlation(d):
    n = len(FEATURES)
    matrix = np.full((n,n), np.nan)
    records = []
    for i,f in enumerate(FEATURES):
        for j,g in enumerate(FEATURES):
            valid = d[[f,g]].notna().all(axis=1) if f != g else d[f].notna()
            rho = 1.0 if f == g else float(spearmanr(d.loc[valid,f], d.loc[valid,g]).statistic)
            matrix[i,j] = rho
            records.append({'feature_row':f,'feature_column':g,'spearman_rho':rho,'n_nodes':int(valid.sum()),
                            'role':'descriptive_no_independent_node_P_values'})
    fig,ax = plt.subplots(figsize=(180*MM,165*MM))
    fig.subplots_adjust(left=.19,right=.88,top=.89,bottom=.19)
    masked = np.ma.masked_where(np.triu(np.ones((n,n)),1).astype(bool), matrix)
    m = ax.imshow(masked, cmap='RdBu_r', vmin=-1,vmax=1)
    for i in range(n):
        for j in range(i+1):
            ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=6.5,
                    color='white' if abs(matrix[i,j])>.75 else 'black')
    ax.set_xticks(range(n), SHORT, rotation=55, ha='right', fontsize=7)
    ax.set_yticks(range(n), SHORT, fontsize=7)
    ax.tick_params(length=0)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    cax=fig.add_axes([.91,.24,.02,.50]); cb=fig.colorbar(m,cax=cax);cb.set_label('Spearman correlation',fontsize=8)
    fig.text(.19,.95,'Recorded node measurements: descriptive Spearman correlations',fontsize=8.5)
    fig.text(.04,.035,'248 nodes in 130 source-recorded groups. Pairwise observed data; n = 247-248.',fontsize=7.5)
    pd.DataFrame(records).to_csv(SRC/'SupplementaryFigure8_correlations.csv',index=False,encoding='utf-8-sig')
    save(fig,'SupplementaryFigure8_Measurement_correlations',
         'Supplementary Figure 8. Correlations among the primary model inputs. Spearman correlations use pairwise observed measurements in the 248 selected nodes from 130 source-recorded groups. One missing effective-atomic-number value gives 247 observations for affected pairs. The matrix is descriptive, without significance testing that would treat correlated nodes as independent. NC and CE denote noncontrast and contrast-enhanced acquisitions. IC denotes iodine concentration, ED electron density relative to water, and Zeff effective atomic number. Full predictor names and pairwise counts accompany the figure.',
         ['SupplementaryFigure8_correlations.csv'])


def distributions(d):
    fig,axes=plt.subplots(4,4,figsize=(180*MM,202*MM))
    fig.subplots_adjust(left=.07,right=.98,top=.93,bottom=.09,hspace=.75,wspace=.48)
    rng=np.random.default_rng(20261008)
    records=[]
    for index,(ax,f) in enumerate(zip(axes.flat,FEATURES)):
        label = SHORT[index] + (f' ({UNITS[f]})' if f in UNITS else '')
        ax.set_title(label,fontsize=7.3,pad=6)
        if f == 'node_rads':
            for cls,offset,color in [(0,-.18,COLORS['NodeRADS']),(1,.18,COLORS['Imaging'])]:
                counts=d.loc[d.outcome==cls,f].value_counts().reindex(range(1,6),fill_value=0)
                ax.bar(np.arange(1,6)+offset,counts.values,width=.33,color=color)
            ax.set_xticks(range(1,6));ax.set_xlabel('Recorded grade',fontsize=7)
            ax.set_ylabel('Node count',fontsize=7)
        else:
            data=[d.loc[d.outcome==cls,f].dropna().to_numpy() for cls in [0,1]]
            ax.boxplot(data,positions=[0,1],widths=.5,showfliers=False,patch_artist=True,
                       boxprops={'facecolor':'white','edgecolor':'black','linewidth':.6},
                       medianprops={'color':'black','linewidth':.8},
                       whiskerprops={'linewidth':.6},capprops={'linewidth':.6})
            for cls,color,values in zip([0,1],[COLORS['NodeRADS'],COLORS['Imaging']],data):
                ax.scatter(cls+rng.uniform(-.17,.17,len(values)),values,s=3,alpha=.30,color=color,zorder=2,lw=0)
            ax.set_xticks([0,1],['Benign','Malignant'],fontsize=6.7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=6.5)
        records.extend({'node_id':r.node_id,'group_id':r.group_id,'outcome':int(r.outcome),'feature':f,'value':r[f]} for _,r in d.iterrows())
    axes.flat[-1].axis('off')
    fig.text(.07,.975,'Distributions of all 15 primary inputs',fontsize=9)
    fig.text(.07,.955,'164 benign and 84 malignant selected nodes; descriptive summaries',fontsize=7.5)
    fig.text(.05,.030,'Points are nodes. Boxes: median and IQR; whiskers: 1.5 x IQR. All observed values remain visible.',fontsize=7)
    pd.DataFrame(records).to_csv(SRC/'SupplementaryFigure9_node_measurements.csv',index=False,encoding='utf-8-sig')
    save(fig,'SupplementaryFigure9_Input_distributions',
         'Supplementary Figure 9. Distributions of all primary model inputs by recorded nodal outcome. Points show observed values for 164 benign and 84 malignant selected nodes. Boxes show the median and interquartile range, with whiskers extending to the most extreme observation within 1.5 times the interquartile range. All observations are plotted, including values beyond the whiskers. The Node-RADS panel shows category counts. One noncontrast effective-atomic-number value is missing. Node measurements are descriptive and are not treated as independent patient replicates for hypothesis testing.',
         ['SupplementaryFigure9_node_measurements.csv'])


def nomogram(d):
    c=pd.read_csv(AN/'final_model_coefficients.csv');c=c[c.model=='Combined'].set_index('feature')
    intercept=float(c.loc['Intercept','coefficient_original_units'])
    beta=c.loc[FEATURES,'coefficient_original_units'].to_numpy()
    ranges=np.array([[d[f].min(),d[f].max()] for f in FEATURES])
    spans=np.abs(beta)*(ranges[:,1]-ranges[:,0])
    factor=100/spans.max()
    baseline=intercept+np.minimum(beta*ranges[:,0],beta*ranges[:,1]).sum()
    totalmax=float((spans*factor).sum())
    data=d[FEATURES].to_numpy(dtype=float,copy=True)
    missing=np.where(np.isnan(data))
    data[missing]=c.loc[FEATURES,'imputation_median'].to_numpy()[missing[1]]
    scores=(beta*data-np.minimum(beta*ranges[:,0],beta*ranges[:,1])).sum(axis=1)*factor
    direct=expit(intercept+data@beta)
    nomop=expit(baseline+scores/factor)
    assert np.max(np.abs(direct-nomop))<1e-12
    fig,ax=plt.subplots(figsize=(180*MM,202*MM))
    fig.subplots_adjust(left=.285,right=.96,top=.93,bottom=.085)
    ax.set_xlim(0,totalmax);ax.set_ylim(-2,19);ax.axis('off')
    xscale=totalmax/100
    records=[]
    def axis(y,pts,labels,name,stagger=False):
        ax.plot([pts.min(),pts.max()],[y,y],color='black',lw=.6)
        ax.text(-.025*totalmax,y,name,ha='right',va='center',fontsize=7.3)
        for index,(x,label) in enumerate(zip(pts,labels)):
            ax.plot([x,x],[y,y-.10],color='black',lw=.5)
            if stagger and index == 0:
                ax.text(x,y+.15,label,ha='center',va='bottom',fontsize=6.5)
            else:
                ax.text(x,y-.22,label,ha='center',va='top',fontsize=6.5)
    axis(18,np.linspace(0,100,6)*xscale,[str(int(v)) for v in np.linspace(0,100,6)],'Points')
    for i,f in enumerate(FEATURES):
        lo,hi=ranges[i]
        pointspan=spans[i]*factor
        vals=np.arange(1,6) if f=='node_rads' else np.linspace(lo,hi,2 if pointspan < 25 else 4)
        vals=np.asarray(vals)
        pts=(beta[i]*vals-min(beta[i]*lo,beta[i]*hi))*factor*xscale
        label=SHORT[i] + (f' ({UNITS[f]})' if f in UNITS else '')
        axis(16.5-i,pts,[f'{v:.3g}' for v in vals],label,stagger=pointspan < 12)
        records.extend({'feature':f,'measurement':float(v),'points':float(p/xscale),'coefficient_original_units':float(beta[i]),'reference_min':float(lo),'reference_max':float(hi)} for v,p in zip(vals,pts))
    totalvalues=np.linspace(0,totalmax,6)
    axis(.0,totalvalues,[f'{v:.0f}' for v in totalvalues],'Total points')
    probabilities=np.array([.01,.10,.5,.90,.99])
    probpoints=(logit(probabilities)-baseline)*factor
    valid=(probpoints>=0)&(probpoints<=totalmax)
    axis(-1.25,probpoints[valid],[f'{p:.2f}' for p in probabilities[valid]],'Node probability')
    fig.text(.04,.975,'Full-development combined ridge model',fontsize=9)
    fig.text(.04,.955,'15 retained inputs; C = 1. Coefficient display for reproduction, without a validated clinical threshold.',fontsize=7.3)
    fig.text(.04,.026,'Nomogram reproduces the saved full-development formula. It does not provide independent validation.',fontsize=7)
    pd.DataFrame(records).to_csv(SRC/'SupplementaryFigure10_nomogram_axes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'node_id':d.node_id,'total_points':scores,'formula_probability':direct,'nomogram_probability':nomop}).to_csv(SRC/'SupplementaryFigure10_nomogram_checks.csv',index=False,encoding='utf-8-sig')
    (SRC/'SupplementaryFigure10_nomogram_formula.json').write_text(json.dumps({'original_unit_intercept':intercept,'log_odds_at_zero_total_points':float(baseline),'points_per_log_odds':float(factor),'max_probability_error':float(np.max(np.abs(direct-nomop))),'measurement_ranges':'full observed ranges in primary cohort; missing values use saved medians'},indent=2),encoding='utf-8')
    save(fig,'SupplementaryFigure10_Combined_model_nomogram',
         'Supplementary Figure 10. Nomogram for the full-development combined ridge model. All 14 direct imaging measurements and the recorded ordinal Node-RADS grade are retained. Each predictor axis covers its observed primary-cohort range. Points are added and converted to the node probability using the original-unit model coefficients. Missing measurements use the saved imputation medians. The mathematical point conversion was verified against the formula for all 248 primary records. This full-development fit is displayed for reproduction. Its in-sample probabilities are not validation estimates, and no clinical classification threshold has been independently validated.',
         ['SupplementaryFigure10_nomogram_axes.csv','SupplementaryFigure10_nomogram_checks.csv','SupplementaryFigure10_nomogram_formula.json'])


def tuning():
    d=pd.read_csv(AN/'inner_tuning_results.csv');s=pd.read_csv(AN/'fitting_details.csv')
    models=['NodeRADS','Imaging','Combined']
    fig,axes=plt.subplots(1,3,figsize=(180*MM,95*MM))
    fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.34,wspace=.34)
    colors=plt.cm.viridis(np.linspace(.12,.88,5))
    for ax,model in zip(axes,models):
        for fold,color in zip(range(1,6),colors):
            g=d[(d.model==model)&(d.outer_fold==fold)].sort_values('C')
            selected=float(s[(s.model==model)&(s.fold==fold)].C.iloc[0])
            val=float(g.loc[np.isclose(g.C,selected),'mean_inner_log_loss'].iloc[0])
            ax.plot(g.C,g.mean_inner_log_loss,color=color,lw=.9,label=f'Outer fold {fold}')
            ax.scatter([selected],[val],s=18,facecolor='white',edgecolor=color,lw=1,zorder=3)
        ax.set_xscale('log');ax.set_xlabel('Inverse ridge penalty C')
        ax.set_title('Node-RADS LR' if model=='NodeRADS' else model,fontsize=8)
    axes[0].set_ylabel('Mean inner-fold log loss')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.53,.11),fontsize=7)
    fig.text(.075,.95,'Grouped inner tuning within each outer development partition',fontsize=8.5)
    fig.text(.075,.035,'Open circles mark selected C. Lower inner log loss determines the penalty; outer outcomes are excluded.',fontsize=7)
    d.to_csv(SRC/'SupplementaryFigure11_inner_tuning.csv',index=False,encoding='utf-8-sig')
    s.to_csv(SRC/'SupplementaryFigure11_selected_settings.csv',index=False,encoding='utf-8-sig')
    save(fig,'SupplementaryFigure11_Ridge_inner_tuning',
         'Supplementary Figure 11. Ridge regularization tuning in the three primary models. Each line shows the mean log loss across three grouped inner folds for one outer development partition. Open circles indicate the selected C from the prespecified grid. Preprocessing was fitted within the corresponding inner training folds. Outer evaluation outcomes did not enter these curves or the selection. Curves describe tuning and are not independent estimates of model performance.',
         ['SupplementaryFigure11_inner_tuning.csv','SupplementaryFigure11_selected_settings.csv'])


if __name__=='__main__':
    setup_style()
    plan={'requested_date':'2026-10-08','primary_analysis_unchanged':True,
          'figures':['Spearman correlations','All input distributions','Full ridge nomogram','Existing grouped tuning traces'],
          'statistical_roles':'Descriptive measurement summaries and model reproduction, without new feature selection or validation claims'}
    (AN/'extended_descriptive_plan_20261008.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    frame=pd.read_csv(AN/'cohort_primary.csv')
    assert len(frame)==248 and frame.group_id.nunique()==130 and frame.outcome.sum()==84
    correlation(frame);distributions(frame);nomogram(frame);tuning()
    (FIG/'extended_descriptive_captions.json').write_text(json.dumps(CAPTIONS,indent=2,ensure_ascii=False),encoding='utf-8')
