"""Exploratory probability calibration and DCA from existing held-out scores.

No estimator fitting, hyperparameter selection, probability transformation, or
threshold optimization is performed. SVM decision margins are not probabilities.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import hashlib
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from pypdf import PdfReader
from make_figures import setup_style
from make_extended_validation_figures import ALGORITHMS, ALG_COLORS, ALG_STYLES

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'figures_tables/source_data'
FIGURE=ROOT/'figures_tables/figures'
SEED=20260926
BOOTSTRAP=2000
MM=1/25.4

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def save(fig,stem,caption,sources):
    fig.canvas.draw()
    for ext in ['pdf','svg','png','tif']:
        settings={'dpi':300,'facecolor':'white'}
        if ext=='tif':settings['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(FIGURE/(stem+'.'+ext),**settings)
    width,height=fig.get_size_inches()/MM
    plt.close(fig)
    png=Image.open(FIGURE/(stem+'.png'));tif=Image.open(FIGURE/(stem+'.tif'))
    assert png.size==tif.size and np.array_equal(np.asarray(png.convert('RGBA')),np.asarray(tif.convert('RGBA')))
    assert np.allclose(tif.info['dpi'],[300,300])
    pdf=PdfReader(FIGURE/(stem+'.pdf'));assert len(pdf.pages)==1
    assert abs(float(pdf.pages[0].mediabox.width)*25.4/72-180)<.01
    assert '<text' in (FIGURE/(stem+'.svg')).read_text(encoding='utf-8')
    return {'stem':stem,'width_mm':float(width),'height_mm':float(height),'dpi':300,'font':'Arial',
            'backend':'Agg','source_files':sources,'caption':caption,'visual_reviewed':False,
            'png_tiff_pixel_equal':True,'pdf_pages':1,'svg_text_editable':True,
            'files':[{ 'file':stem+'.'+ext,'sha256':sha(FIGURE/(stem+'.'+ext))} for ext in ['pdf','svg','png','tif']]}

def main():
    SOURCE.mkdir(parents=True,exist_ok=True);FIGURE.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'analysis/exploratory_predictions.csv',ROOT/'analysis/cohort_primary.csv',
           ROOT/'analysis/primary_predictions.csv',ROOT/'analysis_plan_frozen.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    raw=pd.read_csv(paths[0]);cohort=pd.read_csv(paths[1]).sort_values('node_id').reset_index(drop=True)
    algorithms=[a for a in ALGORITHMS if a!='SVM']
    plan={'operation':'additional_probability_algorithm_figures','plan_created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'input_sha256':hashes,'algorithms':algorithms,'excluded_algorithm':{'SVM':'saved decision_function scores are not probabilities'},
          'n_nodes':248,'n_source_groups':130,'recorded_malignant_nodes':84,
          'model_fits':0,'retuning':False,'new_probability_calibration':False,'new_primary_model_selection':False,
          'bootstrap_replicates':BOOTSTRAP,'bootstrap_seed':SEED,'group_order':'lexical string source-group codes',
          'resampling':'integer group draws with all member nodes and duplicated drawn groups',
          'intervals':'95% percentile conditional on fixed held-out probability scores',
          'dca_thresholds':{'start':.05,'end':.50,'step':.01,'clinical_action_defined':False},
          'calibration_bins':{'quantiles':5,'duplicate_edges':'merge','memberships':'fixed from all held-out scores before group resampling',
                              'zero_weight_bins_in_resample':'omit only that resample for the affected bin and report valid count'},
          'confirmatory_P_values':False,'clinical_benefit_claim':False,
          'figures':['SupplementaryFigure18_Exploratory_algorithm_DCA','SupplementaryFigure19_Exploratory_algorithm_calibration']}
    planpath=ROOT/'analysis/algorithm_probability_figures_plan_before_results.json'
    if planpath.exists():
        prior=json.loads(planpath.read_text(encoding='utf-8'))
        comparison={k:v for k,v in plan.items() if k!='plan_created_at'}
        assert comparison=={k:v for k,v in prior.items() if k!='plan_created_at'}
    else:dump(planpath,plan)
    assert len(cohort)==248 and cohort.group_id.nunique()==130 and int(cohort.outcome.sum())==84
    assert len(raw)==248*11 and set(raw.algorithm)==set(ALGORITHMS)
    assert raw[raw.algorithm=='SVM'].score_type.eq('decision_function').all()
    groups=cohort.group_id.astype(str).to_numpy();unique=sorted(set(groups));lookup={v:i for i,v in enumerate(unique)}
    index=np.array([lookup[g] for g in groups]);rng=np.random.default_rng(SEED)
    groupcounts=np.stack([np.bincount(rng.integers(0,130,size=130),minlength=130) for _ in range(BOOTSTRAP)])
    w=groupcounts[:,index].astype(float);total=w.sum(axis=1);y=cohort.outcome.to_numpy(int)
    assert (total>0).all() and (w@y>0).all() and (w@(1-y)>0).all()
    thresholds=np.round(np.arange(.05,.50001,.01),2)
    prevalence=float(y.mean());treatall=prevalence-(1-prevalence)*thresholds/(1-thresholds)
    dc=[];bc=[];probabilities={};checks=[];brier=[]
    for algorithm in algorithms:
        data=raw[raw.algorithm==algorithm].sort_values('node_id').reset_index(drop=True)
        assert data.node_id.equals(cohort.node_id) and np.array_equal(data.group_id,cohort.group_id)
        assert np.array_equal(data.y_true,y) and np.array_equal(data.fold,cohort.outer_fold)
        assert data.score_type.eq('probability').all() and data.score.between(0,1).all()
        p=data.score.to_numpy(float);probabilities[algorithm]=p
        positive=p[:,None]>=thresholds[None,:]
        tp=(positive*y[:,None]).sum(axis=0);fp=(positive*(1-y[:,None])).sum(axis=0)
        point=tp/248-fp/248*thresholds/(1-thresholds)
        reps=(w@(positive*y[:,None]))/total[:,None]-(w@(positive*(1-y[:,None])))/total[:,None]*thresholds/(1-thresholds)
        limits=np.quantile(reps,[.025,.975],axis=0)
        for draw,thresholdindex in [(0,0),(1999,45),(144,20)]:
            pred=p>=thresholds[thresholdindex]
            direct=(w[draw]*(pred*y)).sum()/total[draw]-(w[draw]*(pred*(1-y))).sum()/total[draw]*thresholds[thresholdindex]/(1-thresholds[thresholdindex])
            assert np.isclose(direct,reps[draw,thresholdindex],atol=1e-12)
        for j,t in enumerate(thresholds):
            dc.append({'algorithm':algorithm,'threshold':t,'net_benefit':point[j],'ci_lower':limits[0,j],'ci_upper':limits[1,j],
                       'treat_all':treatall[j],'treat_none':0.,'true_positives':int(tp[j]),'false_positives':int(fp[j]),
                       'n_nodes':248,'n_source_groups':130,'bootstrap_replicates':BOOTSTRAP,'bootstrap_seed':SEED,
                       'interval_method':'fixed_probability_source_group_percentile_bootstrap','analysis_role':'exploratory_no_clinical_action_defined'})
        bins=pd.qcut(p,q=5,labels=False,duplicates='drop')
        assert np.isfinite(bins).all()
        score=float(np.mean((p-y)**2));brier.append({'algorithm':algorithm,'brier_score':score,'n_nodes':248,'n_source_groups':130,
                                                   'analysis_role':'exploratory_fixed_heldout_probabilities'})
        for label in np.unique(bins):
            member=bins==label;n=int(member.sum());denominator=w@member.astype(float)
            observed=(w@(member*y))/np.where(denominator==0,1,denominator);valid=denominator>0
            ci=np.quantile(observed[valid],[.025,.975]);events=int(y[member].sum())
            for draw in [0,144,1999]:
                if denominator[draw]>0:
                    direct=np.sum(w[draw,member]*y[member])/np.sum(w[draw,member])
                    assert np.isclose(direct,observed[draw],atol=1e-12)
            bc.append({'algorithm':algorithm,'bin':int(label)+1,'n_nodes':n,'n_source_groups':int(len(set(groups[member]))),
                       'malignant_nodes':events,'mean_predicted_probability':float(p[member].mean()),'observed_fraction':events/n,
                       'ci_lower':ci[0],'ci_upper':ci[1],'minimum_probability':float(p[member].min()),'maximum_probability':float(p[member].max()),
                       'bootstrap_valid':int(valid.sum()),'bootstrap_replicates':BOOTSTRAP,'bootstrap_seed':SEED,
                       'fixed_bin_membership':True,'analysis_role':'exploratory_no_recalibration_or_model_selection'})
        checks.append({'algorithm':algorithm,'n_nodes':248,'same_node_group_outcome_fold_alignment':True,
                       'probability_in_unit_interval':True,'DCA_point_and_selected_draws_verified':True,
                       'calibration_bins':int(len(np.unique(bins))),'calibration_bin_count_sum':int(sum((bins==k).sum() for k in np.unique(bins))),
                       'calibration_selected_draws_verified':True,'brier_score':score})
    dca=pd.DataFrame(dc);calibration=pd.DataFrame(bc);bs=pd.DataFrame(brier)
    dca.to_csv(SOURCE/'SupplementaryFigure18_algorithm_DCA_source.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    calibration.to_csv(SOURCE/'SupplementaryFigure19_algorithm_calibration_bins.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    bs.to_csv(SOURCE/'SupplementaryFigure19_algorithm_Brier_source.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    raw[raw.algorithm.isin(algorithms)].to_csv(SOURCE/'SupplementaryFigure18_19_heldout_probability_source.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    setup_style();plt.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7})
    fig,axes=plt.subplots(5,2,figsize=(180*MM,257*MM),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.93,hspace=.67,wspace=.29)
    fig.suptitle('Exploratory probability algorithms: held-out decision curves',x=.02,y=.985,ha='left',fontsize=9)
    ymin=np.floor(min(dca.ci_lower.min(),treatall.min())/.1)*.1;ymax=np.ceil(max(dca.ci_upper.max(),treatall.max())/.1)*.1
    for i,(algorithm,ax) in enumerate(zip(algorithms,axes.ravel())):
        idx=ALGORITHMS.index(algorithm);data=dca[dca.algorithm==algorithm]
        ax.fill_between(data.threshold,data.ci_lower,data.ci_upper,color=ALG_COLORS[idx],alpha=.16,linewidth=0)
        ax.plot(data.threshold,data.net_benefit,color=ALG_COLORS[idx],linestyle=ALG_STYLES[idx],linewidth=1.05)
        ax.plot(thresholds,treatall,color='#777777',linestyle='--',linewidth=.85)
        ax.axhline(0,color='black',linestyle=':',linewidth=.85)
        ax.set_title(chr(65+i)+'. '+algorithm,loc='left',pad=5)
        ax.set_xlim(.05,.50);ax.set_ylim(ymin,ymax);ax.set_xticks([.1,.2,.3,.4,.5]);ax.set_yticks(np.arange(np.ceil(ymin/.2)*.2,ymax+.0001,.2))
        if i%2==0:ax.set_ylabel('Net benefit')
        if i>=8:ax.set_xlabel('Threshold probability')
        ax.tick_params(length=2);ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([],[],color='black',label='Named algorithm (colored curve)'),Line2D([],[],color='#777777',linestyle='--',label='Treat all'),Line2D([],[],color='black',linestyle=':',label='Treat none')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,.069),ncol=3,frameon=False,fontsize=7)
    fig.text(.055,.048,'248 nodes in 130 source-recorded groups. Shading: 95% CIs from 2,000 fixed-score group resamples.',fontsize=7)
    fig.text(.055,.025,'The 0.05-0.50 range is exploratory. No clinical action is specified. SVM margins are excluded.',fontsize=7)
    cap1='Supplementary Figure 18. Exploratory decision curves for the ten imaging algorithms with saved probability scores. Each panel uses the same 248 held-out nodes in 130 source-recorded groups, including 84 recorded malignant nodes. Colored curves are the named algorithms, gray dashed curves Treat all, and black dotted lines Treat none. Net benefit is TP/n minus FP/n times threshold/(1 minus threshold). Shading contains 95% percentile intervals from 2,000 shared source-group bootstrap resamples conditional on fixed held-out probabilities. The threshold range 0.05 to 0.50 has no specified clinical action and does not establish surgical benefit. SVM decision_function scores are excluded because they are not probabilities. These algorithms remain exploratory and do not select or replace the primary ridge models. Hospital patient identity linkage remains unverified.'
    manifest=[save(fig,'SupplementaryFigure18_Exploratory_algorithm_DCA',cap1,['SupplementaryFigure18_algorithm_DCA_source.csv','SupplementaryFigure18_19_heldout_probability_source.csv'])]
    fig,axes=plt.subplots(5,2,figsize=(180*MM,257*MM),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.93,hspace=.65,wspace=.29)
    fig.suptitle('Exploratory probability algorithms: held-out calibration',x=.02,y=.985,ha='left',fontsize=9)
    for i,(algorithm,ax) in enumerate(zip(algorithms,axes.ravel())):
        idx=ALGORITHMS.index(algorithm);data=calibration[calibration.algorithm==algorithm]
        ax.plot([0,1],[0,1],color='#777777',linestyle='--',linewidth=.8)
        error=np.vstack([data.observed_fraction-data.ci_lower,data.ci_upper-data.observed_fraction])
        ax.errorbar(data.mean_predicted_probability,data.observed_fraction,yerr=error,fmt='o-',color=ALG_COLORS[idx],
                    markersize=3.0,capsize=2,linewidth=1.0,elinewidth=.8)
        point=float(bs[bs.algorithm==algorithm].brier_score.iloc[0])
        ax.set_title(chr(65+i)+'. '+algorithm+f'   Brier {point:.3f}',loc='left',pad=5)
        ax.set_xlim(-.025,1.025);ax.set_ylim(-.025,1.025);ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1])
        if i%2==0:ax.set_ylabel('Observed fraction')
        if i>=8:ax.set_xlabel('Mean predicted probability')
        ax.tick_params(length=2);ax.spines[['top','right']].set_visible(False)
    fig.text(.055,.071,'248 nodes in 130 source-recorded groups. Up to five quantile bins; duplicate boundaries are merged.',fontsize=7)
    fig.text(.055,.048,'Vertical bars: observed-fraction 95% CIs from 2,000 shared group resamples with fixed bin membership.',fontsize=7)
    fig.text(.055,.025,'Brier scores use the same held-out nodes. No recalibration or algorithm selection. SVM margins are excluded.',fontsize=7)
    cap2='Supplementary Figure 19. Exploratory calibration of the ten imaging algorithms with saved probability scores. Each panel uses the same 248 held-out nodes in 130 source-recorded groups. Points are means of predicted probabilities and recorded malignant fractions in up to five quantile bins, with duplicate boundaries merged. Vertical bars are observed-fraction 95% percentile source-group bootstrap intervals conditional on fixed probability scores and fixed original bin memberships. Resamples with zero members in a given bin are omitted only for that bin and its valid replicate count is reported in source data. Dashed lines indicate ideal calibration. Panel Brier scores are node-level point estimates from the same held-out probabilities. No probability recalibration, algorithm retuning, or primary-model selection occurs. SVM decision_function scores are excluded because they are not probabilities. Bin sizes and malignant counts are in the source CSV. These are exploratory internal results and do not establish external validation or calibrated whole-patient pN probability.'
    manifest.append(save(fig,'SupplementaryFigure19_Exploratory_algorithm_calibration',cap2,['SupplementaryFigure19_algorithm_calibration_bins.csv','SupplementaryFigure19_algorithm_Brier_source.csv','SupplementaryFigure18_19_heldout_probability_source.csv']))
    for name,checksum in hashes.items():assert sha(ROOT/name)==checksum
    dump(FIGURE/'algorithm_probability_figure_captions.json',[{'title':m['caption'].split('. ',2)[0]+'. '+m['caption'].split('. ',2)[1],
                                                             'text':m['caption'].split('. ',2)[2],'path':str(FIGURE/(m['stem']+'.png')),'supplementary':True} for m in manifest])
    dump(FIGURE/'algorithm_probability_figures_QA.json',{'input_sha256':hashes,'new_model_fits':0,'algorithm_selection':False,
        'algorithms':algorithms,'excluded_SVM_margin':True,'seed':SEED,'bootstrap_replicates':BOOTSTRAP,
        'group_resamples_shared_across_algorithms_and_figures':True,'numeric_checks':checks,'figures':manifest,
        'visual_review_status':'Awaiting individual PNG review','dca_rows':len(dca),'calibration_bins':len(calibration)})
    print(json.dumps({'new_figures':len(manifest),'probability_algorithms':len(algorithms),'DCA_rows':len(dca),
                      'calibration_bins':len(calibration),'minimum_valid_calibration_bootstrap':int(calibration.bootstrap_valid.min()),'model_fits':0}))

if __name__=='__main__':main()
