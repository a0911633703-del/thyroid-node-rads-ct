"""Extended validation figures using completed held-out predictions only.

No estimator fit, retuning or threshold selection occurs. Run with the recorded Python environment.
S4 pooled ROC/AP intervals use a fixed-score source-group bootstrap, without model refitting.
"""
from __future__ import annotations

import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score

from make_figures import setup_style, COLORS, MARKERS, MODELS

ROOT=Path(__file__).resolve().parents[1]
MM=1/25.4
LABEL={'NodeRADS':'Node-RADS LR','Imaging':'Imaging parameters','Combined':'Combined'}
ALGORITHMS=['AdaBoost','Extra Trees','Gradient boosting','KNN','LightGBM','Logistic regression',
            'MLP','Naive Bayes','Random forest','SVM','XGBoost']
ALG_LABEL={'Logistic regression':'Logistic regression','Gradient boosting':'Gradient boosting',
           'Random forest':'Random forest','Naive Bayes':'Naive Bayes'}
ALG_COLORS=['#CC79A7','#7F3C8D','#A6761D','#56B4E9','#009E73','#D55E00',
            '#E69F00','#8C8C8C','#0072B2','#332288','#000000']
ALG_STYLES=['-','--','-.','-','--','-','-.',':','-','--','-.']


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def clustered_score_intervals(y,score,group_weights):
    """Vectorized weighted ROC-AUC/AP, retaining score ties exactly."""
    order=np.argsort(score,kind='stable')
    sorted_score=score[order];ys=y[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_score)!=0),len(score)-1]
    w=group_weights[:,order]
    pos=np.cumsum(w*ys[None,:],axis=1)[:,ends]
    neg=np.cumsum(w*(1-ys)[None,:],axis=1)[:,ends]
    pos_group=np.diff(pos,axis=1,prepend=0)
    neg_group=np.diff(neg,axis=1,prepend=0)
    npos=pos[:,-1];nneg=neg[:,-1]
    auc=np.sum(pos_group*(neg-.5*neg_group),axis=1)/(npos*nneg)
    positives_desc=pos_group[:,::-1];negatives_desc=neg_group[:,::-1]
    tp=np.cumsum(positives_desc,axis=1);total=np.cumsum(positives_desc+negatives_desc,axis=1)
    precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    ap=np.sum(precision*positives_desc/npos[:,None],axis=1)
    assert np.isfinite(auc).all() and np.isfinite(ap).all()
    return auc,ap


class ExtendedFigures:
    def __init__(self,root,bootstrap):
        self.root=root;self.an=root/'analysis';self.figdir=root/'figures_tables/figures'
        self.srcdir=root/'figures_tables/source_data';self.bootstrap=bootstrap
        self.figdir.mkdir(parents=True,exist_ok=True);self.srcdir.mkdir(parents=True,exist_ok=True)
        self.manifest=[];self.input_hashes={};self.numeric_checks=[]

    def read(self,name):
        p=self.an/name
        self.input_hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        return pd.read_csv(p)

    def source(self,data,name):
        data.to_csv(self.srcdir/name,index=False,encoding='utf-8-sig',float_format='%.12g')

    def save(self,fig,stem,caption,sources):
        fig.canvas.draw()
        assert np.isclose(fig.get_size_inches()[0]/MM,180)
        for ext in ['pdf','svg','png','tif']:
            kwargs={'dpi':300,'facecolor':'white'}
            if ext=='tif':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
            fig.savefig(self.figdir/f'{stem}.{ext}',**kwargs)
        self.manifest.append({'figure':stem,'width_mm':180,'height_mm':float(fig.get_size_inches()[1]/MM),
                              'dpi':300,'backend':'Agg','font':'Arial','source_files':sources,'caption':caption,
                              'files':[f'{stem}.{ext}' for ext in ['pdf','svg','png','tif']],
                              'visual_reviewed':False})
        plt.close(fig)
        print('Saved',stem,flush=True)

    def s3_confusion(self):
        pred=self.read('primary_predictions.csv');records=[]
        fig=plt.figure(figsize=(180*MM,87*MM))
        fig.text(.02,.98,'Classification at development-selected thresholds',ha='left',va='top',fontsize=9)
        for idx,model in enumerate(MODELS):
            d=pred[pred.model==model]
            assert len(d)==248 and d.node_id.nunique()==248
            assert np.array_equal(d.prediction,d.probability.ge(d.threshold).astype(int))
            matrix=np.zeros((2,2),dtype=int)
            for recorded in [0,1]:
                for predicted in [0,1]:
                    count=int(((d.y_true==recorded)&(d.prediction==predicted)).sum())
                    matrix[recorded,predicted]=count
                    total=int((d.y_true==recorded).sum())
                    records.append({'model':model,'recorded_outcome':recorded,'predicted_outcome':predicted,
                                    'count':count,'recorded_class_total':total,'percentage_within_recorded_class':100*count/total,
                                    'n_nodes':len(d),'n_source_groups':d.group_id.nunique(),
                                    'threshold_scope':'fold_specific_inner_development_only'})
            pct=100*matrix/matrix.sum(axis=1,keepdims=True)
            ax=fig.add_axes([.075+idx*.325,.265,.25,.57])
            cmap=LinearSegmentedColormap.from_list('pastel_'+model,['white',COLORS[model]])
            ax.imshow(pct,cmap=cmap,vmin=0,vmax=360,interpolation='nearest',aspect='equal')
            for r in [0,1]:
                for c in [0,1]:
                    kind=[['TN','FP'],['FN','TP']][r][c]
                    ax.text(c,r,f'{kind}: {matrix[r,c]}\n{pct[r,c]:.1f}%',ha='center',va='center',fontsize=8,color='black',linespacing=1.5)
            ax.set_xticks([0,1],['Predicted\nbenign','Predicted\nmalignant'])
            ax.set_yticks([0,1],['Recorded\nbenign','Recorded\nmalignant'])
            ax.tick_params(length=0,labelsize=7)
            ax.set_xticks(np.arange(-.5,2,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True)
            ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',length=0)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.set_title(LABEL[model],fontsize=8,pad=12)
            fig.text(.025+idx*.325,.88,chr(65+idx),fontsize=10,fontweight='bold')
            self.numeric_checks.append({'figure':'S3','model':model,'TN_FP_FN_TP':matrix.flatten().tolist()})
        fig.text(.055,.12,'248 nodes: 164 recorded benign and 84 recorded malignant. Percentages use each recorded outcome as the denominator.',fontsize=7,ha='left')
        fig.text(.055,.055,'Each classification uses its fold-specific inner-development threshold; outer outcomes do not select thresholds.',fontsize=7,ha='left')
        source='SupplementaryFigure3_confusion_matrix_source.csv';self.source(pd.DataFrame(records),source)
        self.save(fig,'SupplementaryFigure3_Confusion_matrices',
                  'Supplementary Figure 3. Classification counts at development-selected thresholds. '
                  'The three primary ridge logistic models use the same 248 held-out nodes in 130 source-recorded groups. '
                  'Rows are recorded benign and malignant outcomes, columns predicted outcomes. Cells show counts and '
                  'percentages within the recorded outcome class. Classification uses each outer fold\'s threshold selected '
                  'from inner out-of-fold development predictions. TN, true negative. FP, false positive. FN, false negative. TP, true positive. '
                  'Color intensity also represents within-class percentage.',[source])

    def s4_algorithms(self):
        allpred=self.read('exploratory_predictions.csv');cohort=self.read('cohort_primary.csv')
        assert len(allpred)==248*11 and set(allpred.algorithm)==set(ALGORITHMS)
        baseline=cohort[['node_id','group_id','outcome','outer_fold']].sort_values('node_id').reset_index(drop=True)
        assert baseline.node_id.nunique()==248 and baseline.group_id.nunique()==130
        groups=sorted(baseline.group_id.unique());group_index={g:i for i,g in enumerate(groups)}
        indexes=np.array([group_index[g] for g in baseline.group_id])
        rng=np.random.default_rng(20260926)
        draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=self.bootstrap)
        weights=draws[:,indexes].astype(float)
        y=baseline.outcome.to_numpy();prevalence=float(y.mean())
        fig=plt.figure(figsize=(180*MM,173*MM))
        axes=[fig.add_axes([.105,.475,.365,.405]),fig.add_axes([.595,.475,.365,.405])]
        fig.text(.02,.985,'Exploratory algorithms: pooled held-out discrimination',fontsize=9,ha='left',va='top')
        records=[];coords=[]
        for i,algorithm in enumerate(ALGORITHMS):
            d=allpred[allpred.algorithm==algorithm].sort_values('node_id').reset_index(drop=True)
            assert d.node_id.equals(baseline.node_id)
            assert np.array_equal(d.group_id,baseline.group_id) and np.array_equal(d.y_true,baseline.outcome)
            assert np.array_equal(d.fold,baseline.outer_fold)
            score=d.score.to_numpy(float)
            if algorithm=='SVM':assert d.score_type.eq('decision_function').all()
            else:assert d.score_type.eq('probability').all() and d.score.between(0,1).all()
            auc=float(roc_auc_score(y,score));ap=float(average_precision_score(y,score))
            # Independently compare vectorized point estimates to sklearn before bootstrap.
            point_auc,point_ap=clustered_score_intervals(y,score,np.ones((1,len(y))))
            assert np.isclose(point_auc[0],auc,atol=1e-12) and np.isclose(point_ap[0],ap,atol=1e-12)
            boot_auc,boot_ap=clustered_score_intervals(y,score,weights)
            auc_ci=np.quantile(boot_auc,[.025,.975]);ap_ci=np.quantile(boot_ap,[.025,.975])
            fpr,tpr,rt=roc_curve(y,score);precision,recall,pt=precision_recall_curve(y,score)
            color,style=ALG_COLORS[i],ALG_STYLES[i]
            name=ALG_LABEL.get(algorithm,algorithm)
            for ax,x,z,value,ci,metric in [(axes[0],fpr,tpr,auc,auc_ci,'AUC'),(axes[1],recall,precision,ap,ap_ci,'AP')]:
                ax.plot(x,z,color=color,linestyle=style,linewidth=1.1,alpha=.92,
                        label=f'{name}: {value:.3f} ({ci[0]:.3f}-{ci[1]:.3f})')
            records.append({'algorithm':algorithm,'n_nodes':248,'n_source_groups':130,'malignant_nodes':84,
                            'pooled_roc_auc':auc,'auc_ci_lower':float(auc_ci[0]),'auc_ci_upper':float(auc_ci[1]),
                            'pooled_average_precision':ap,'ap_ci_lower':float(ap_ci[0]),'ap_ci_upper':float(ap_ci[1]),
                            'bootstrap_replicates':self.bootstrap,'bootstrap_seed':20260926,
                            'score_type':d.score_type.iloc[0],
                            'interval_method':'fixed_heldout_scores_source_group_percentile_bootstrap',
                            'analysis_role':'exploratory_pooled_scores_no_model_refit_no_pairwise_tests'})
            for x,z,threshold in zip(fpr,tpr,rt):
                coords.append({'algorithm':algorithm,'curve':'ROC','x':x,'y':z,'threshold':threshold if np.isfinite(threshold) else np.nan})
            for j,(x,z) in enumerate(zip(recall,precision)):
                coords.append({'algorithm':algorithm,'curve':'PR','x':x,'y':z,'threshold':pt[j] if j<len(pt) else np.nan})
        for idx,ax in enumerate(axes):
            ax.set_xlim(0,1);ax.set_ylim(0,1.01)
            ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
            ax.tick_params(labelsize=7)
            ax.legend(loc='upper left',bbox_to_anchor=(-.025,-.17),frameon=False,fontsize=6.6,
                      handlelength=2.2,labelspacing=.5,borderaxespad=0,handletextpad=.45)
            fig.text(.055+idx*.49,.92,chr(65+idx),fontsize=10,fontweight='bold')
        axes[0].plot([0,1],[0,1],color='#929292',linestyle=':',linewidth=.9)
        axes[0].set_xlabel('1 - specificity');axes[0].set_ylabel('Sensitivity');axes[0].set_title('Receiver operating characteristic',fontsize=8)
        axes[1].axhline(prevalence,color='#929292',linestyle=':',linewidth=.9)
        axes[1].set_xlabel('Recall (sensitivity)');axes[1].set_ylabel('Precision');axes[1].set_title('Precision-recall',fontsize=8)
        axes[1].text(.02,prevalence+.035,f'Sample prevalence = {prevalence:.3f}',fontsize=7,color='#555555')
        fig.text(.065,.035,'AUC and AP use pooled held-out scores, rather than mean fold estimates. CIs use 2,000 source-group resamples.',fontsize=7,ha='left')
        fig.text(.065,.012,'Comparisons are exploratory. SVM uses decision scores. Algorithms and tuning are fixed; no primary model is reselected.',fontsize=7,ha='left')
        metrics_file='SupplementaryFigure4_pooled_algorithm_metrics.csv';coords_file='SupplementaryFigure4_ROC_PR_coordinates.csv'
        predictions_file='SupplementaryFigure4_heldout_algorithm_scores.csv'
        self.source(pd.DataFrame(records),metrics_file);self.source(pd.DataFrame(coords),coords_file);self.source(allpred,predictions_file)
        self.numeric_checks.append({'figure':'S4','algorithms':11,'predictions':len(allpred),'point_estimates_match_sklearn':True,
                                    'group_bootstrap_replicates':self.bootstrap,'SVM_probability_conversion':False})
        self.save(fig,'SupplementaryFigure4_Exploratory_ROC_PR',
                  'Supplementary Figure 4. Pooled held-out ROC and precision-recall curves for 11 exploratory algorithms. '
                  'All algorithms use the same 248 nodes, 130 source-recorded groups, 14 imaging inputs and retained outer folds. '
                  'Legends display ROC-AUC or average precision with 95% fixed-score percentile intervals from 2,000 source-group bootstrap samples. '
                  'Average precision is the noninterpolated PR summary. These pooled-score summaries differ from the mean fold AUCs in '
                  'Supplementary Figure 1 and Supplementary Table 5. The ROC diagonal denotes no discrimination. The PR reference is the selected-node malignant fraction, '
                  '84/248. SVM uses its decision score without converting it to a probability. No model refitting, algorithm reselection or pairwise significance testing was performed.',
                  [metrics_file,coords_file,predictions_file])

    def s5_sensitivity(self):
        primary=self.read('metrics_primary.csv').copy();primary['scenario']='primary'
        sens=self.read('sensitivity_metrics_summary.csv')
        allmetrics=pd.concat([primary,sens],ignore_index=True)
        scenarios=[('primary','Primary (>3 mm)'),('all_254','Complete source table'),('legacy_249','Original 249-node set'),
                   ('reference_augmented','Add reference attenuation'),('normalized_CT','CT ratios and diameters'),('exclude_group_76','Exclude source group 76')]
        fig=plt.figure(figsize=(180*MM,127*MM));ax=fig.add_axes([.36,.235,.32,.64]);bx=fig.add_axes([.76,.235,.215,.64])
        fig.text(.02,.985,'Sensitivity to cohort and predictor definitions',fontsize=9,ha='left',va='top')
        fig.text(.315,.91,'A',fontsize=10,fontweight='bold');fig.text(.707,.91,'B',fontsize=10,fontweight='bold')
        ax.set_title('ROC-AUC for all three models',fontsize=8);bx.set_title('Combined-model Brier score',fontsize=8)
        rows=[];labels=[]
        for i,(scenario,label) in enumerate(scenarios):
            d=allmetrics[allmetrics.scenario==scenario]
            count=d.iloc[0]
            labels.append(label+f'\nn = {int(count.n_nodes)}, M = {int(count.malignant_nodes)}, G = {int(count.n_source_groups)}')
            if i%2==0:
                ax.axhspan(i-.46,i+.46,color='#F4F5F6',zorder=0)
                bx.axhspan(i-.46,i+.46,color='#F4F5F6',zorder=0)
            for j,model in enumerate(MODELS):
                r=d[(d.model==model)&(d.metric=='roc_auc')].iloc[0]
                y=i+(j-1)*.23
                ax.errorbar(r.estimate,y,xerr=np.array([[r.estimate-r.ci_lower],[r.ci_upper-r.estimate]]),
                            fmt=MARKERS[model],color=COLORS[model],markersize=4,capsize=2,linewidth=1,elinewidth=1,zorder=2)
                rows.append(r.to_dict())
            r=d[(d.model=='Combined')&(d.metric=='brier_score')].iloc[0]
            bx.errorbar(r.estimate,i-.13,xerr=np.array([[r.estimate-r.ci_lower],[r.ci_upper-r.estimate]]),
                        fmt=MARKERS['Combined'],color=COLORS['Combined'],markersize=4,capsize=2,linewidth=1,elinewidth=1,zorder=2)
            bx.text(.077,i+.27,f'{r.estimate:.3f} ({r.ci_lower:.3f}-{r.ci_upper:.3f})',ha='center',va='center',fontsize=6.8)
            rows.append(r.to_dict())
        for axes in [ax,bx]:
            axes.set_ylim(5.5,-.7);axes.set_yticks(range(6));axes.tick_params(axis='y',length=0)
            axes.spines['left'].set_visible(False)
        ax.set_yticklabels(labels,fontsize=7.1);bx.set_yticklabels([])
        ax.set_xlim(.765,1.0);ax.set_xticks([.8,.85,.9,.95,1]);ax.tick_params(axis='x',labelsize=7)
        bx.set_xlim(.04,.114);bx.set_xticks([.04,.06,.08,.1]);bx.tick_params(axis='x',labelsize=7)
        ax.set_xlabel('ROC-AUC (95% CI)');bx.set_xlabel('Brier score (95% CI)')
        handles=[Line2D([],[],marker=MARKERS[m],color=COLORS[m],linestyle='-',label=LABEL[m],markersize=4) for m in MODELS]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.125),ncol=3,frameon=False,fontsize=7)
        fig.text(.055,.092,'n = selected nodes; M = malignant nodes; G = source-recorded groups. All scenarios retain the same group-fold allocation.',fontsize=7)
        fig.text(.055,.055,'95% intervals condition on fixed held-out predictions within each scenario. Different eligibility or inputs are exploratory.',fontsize=7)
        fig.text(.055,.025,'The original 249-node set includes nodes ≤3 mm and omits five zero-iodine benign nodes; it is not a separate validation cohort.',fontsize=7)
        source='SupplementaryFigure5_sensitivity_performance.csv';self.source(pd.DataFrame(rows),source)
        self.numeric_checks.append({'figure':'S5','scenarios':6,'auc_estimates':18,'combined_brier_estimates':6,
                                    'interval_source':'existing primary and sensitivity fixed-oof cluster-bootstrap outputs'})
        self.save(fig,'SupplementaryFigure5_Sensitivity_performance',
                  'Supplementary Figure 5. Performance under the primary definition and five sensitivity scenarios. '
                  'A, pooled held-out ROC-AUC with 95% source-group bootstrap intervals for the Node-RADS logistic, imaging and combined models. '
                  'B, combined-model Brier scores and corresponding intervals. Counts identify selected nodes (n), malignant nodes (M) and source-recorded groups (G). '
                  'Outer-fold group allocation is shared across scenarios. The reference-augmented set adds raw arterial and muscle attenuation. '
                  'The CT-ratio imaging set contains node-to-artery and node-to-muscle CT ratios and both diameters, with Node-RADS added in its combined model. '
                  'The historical subset is its original membership, which includes nodes failing the size criterion and excludes five benign zero-iodine nodes. '
                  'Intervals condition on each scenario\'s fixed held-out predictions. Sensitivity scenarios are exploratory, do not change the primary model, and do not constitute external validation.',[source])

    def s3b_s4c_default_classification(self):
        allpred=self.read('exploratory_predictions.csv');cohort=self.read('cohort_primary.csv')
        baseline=cohort.sort_values('node_id').reset_index(drop=True)
        groups=sorted(baseline.group_id.unique());mapping={g:i for i,g in enumerate(groups)}
        indexes=np.array([mapping[g] for g in baseline.group_id])
        rng=np.random.default_rng(20260926)
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=self.bootstrap)[:,indexes]
        classification=[];folds=[];metrics=[];confusion=[]
        fig=plt.figure(figsize=(180*MM,192*MM))
        fig.text(.02,.985,'Exploratory algorithms: held-out classification at default thresholds',fontsize=9,ha='left',va='top')
        for i,algorithm in enumerate(ALGORITHMS):
            d=allpred[allpred.algorithm==algorithm].sort_values('node_id').reset_index(drop=True)
            assert d.node_id.equals(baseline.node_id) and np.array_equal(d.y_true,baseline.outcome)
            threshold=0 if algorithm=='SVM' else .5
            assert not np.isclose(d.score,threshold,rtol=0,atol=0).any()
            predicted=d.score.ge(threshold).astype(int).to_numpy();y=d.y_true.to_numpy()
            correct=predicted==y;point=float(correct.mean())
            boot=(weights@correct.astype(float))/weights.sum(axis=1);ci=np.quantile(boot,[.025,.975])
            matrix=np.array([[int(((y==r)&(predicted==c)).sum()) for c in [0,1]] for r in [0,1]])
            pct=100*matrix/matrix.sum(axis=1,keepdims=True)
            metrics.append({'algorithm':algorithm,'n_nodes':248,'n_source_groups':130,'malignant_nodes':84,
                            'pooled_accuracy':point,'ci_lower':ci[0],'ci_upper':ci[1],
                            'default_threshold':threshold,'score_type':d.score_type.iloc[0],
                            'threshold_scope':'default_0.5_probability_or_0_SVM_decision_not_inner_Youden',
                            'bootstrap_replicates':self.bootstrap,'bootstrap_seed':20260926,
                            'interval_method':'fixed_heldout_classification_source_group_percentile_bootstrap'})
            for f in range(1,6):
                mask=d.fold.eq(f).to_numpy();folds.append({'algorithm':algorithm,'fold':f,'n_nodes':int(mask.sum()),
                                                          'correct_nodes':int(correct[mask].sum()),'accuracy':float(correct[mask].mean()),
                                                          'default_threshold':threshold})
            for n,row in d.iterrows():
                classification.append({'algorithm':algorithm,'fold':int(row.fold),'node_id':row.node_id,'group_id':int(row.group_id),
                                       'y_true':int(row.y_true),'score':float(row.score),'predicted':int(predicted[n]),
                                       'threshold':threshold,'score_type':row.score_type})
            for r in [0,1]:
                for c in [0,1]:
                    confusion.append({'algorithm':algorithm,'recorded_outcome':r,'predicted_outcome':c,'count':int(matrix[r,c]),
                                      'percentage_within_recorded_class':float(pct[r,c]),'recorded_class_total':int(matrix[r].sum()),
                                      'threshold':threshold,'score_type':d.score_type.iloc[0]})
            row,col=divmod(i,4)
            ax=fig.add_axes([.07+col*.245,.69-row*.285,.18,.215])
            cmap=LinearSegmentedColormap.from_list('pastel_alg'+str(i),['white',ALG_COLORS[i]])
            ax.imshow(pct,cmap=cmap,vmin=0,vmax=360,interpolation='nearest',aspect='equal')
            for r in [0,1]:
                for c in [0,1]:ax.text(c,r,f'{[["TN","FP"],["FN","TP"]][r][c]} {matrix[r,c]}\n{pct[r,c]:.1f}%',ha='center',va='center',fontsize=6.8,linespacing=1.5)
            ax.set_xticks([0,1],['Pred. benign','Pred. malignant']);ax.set_yticks([0,1],['Benign','Malignant'])
            ax.tick_params(length=0,labelsize=6.5)
            ax.set_xticks(np.arange(-.5,2,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True)
            ax.grid(which='minor',color='white',linewidth=1.3);ax.tick_params(which='minor',length=0)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.set_title(algorithm+f'\nAccuracy {100*point:.1f}%',fontsize=7,pad=6)
            if col==0:ax.set_ylabel('Recorded outcome',fontsize=6.5)
        blank=fig.add_axes([.805,.12,.18,.215]);blank.axis('off')
        blank.text(.02,.96,'Fixed held-out scores\n\nProbability ≥0.5\nSVM decision score ≥0\n\n248 selected nodes\n130 recorded groups\n\nNo threshold tuning',ha='left',va='top',fontsize=7,linespacing=1.4)
        fig.text(.055,.065,'Percentages use each recorded outcome as the denominator. Default thresholds differ from the primary models’ inner Youden thresholds.',fontsize=6.8)
        fig.text(.055,.032,'All classifications are held out. No apparent training accuracy is reported; algorithm and threshold comparisons are descriptive.',fontsize=6.8)
        confusion_file='SupplementaryFigure3_exploratory_confusion_source.csv'
        predictions_file='SupplementaryFigure3_exploratory_classification_source.csv'
        self.source(pd.DataFrame(confusion),confusion_file);self.source(pd.DataFrame(classification),predictions_file)
        self.save(fig,'SupplementaryFigure3B_Exploratory_confusion_matrices',
                  'Supplementary Figure 3B. Held-out confusion matrices for the 11 exploratory algorithms. '
                  'Every algorithm uses the same 248 nodes and retained recorded-group outer folds. Probability models are classified '
                  'at 0.5 and SVM decision scores at 0. Cells show counts and percentages within recorded benign or malignant outcomes. '
                  'The thresholds are classifier defaults, not inner Youden thresholds used in the primary models. '
                  'No apparent training accuracy or new threshold optimization is reported.',[confusion_file,predictions_file])
        summaries=pd.DataFrame(metrics);folddata=pd.DataFrame(folds)
        summaries=summaries.assign(mean_fold_accuracy=[folddata[folddata.algorithm==a].accuracy.mean() for a in summaries.algorithm],
                                    sd_fold_accuracy=[folddata[folddata.algorithm==a].accuracy.std(ddof=1) for a in summaries.algorithm])
        fig=plt.figure(figsize=(180*MM,137*MM));ax=fig.add_axes([.20,.26,.31,.62]);bx=fig.add_axes([.63,.26,.34,.62])
        fig.text(.02,.985,'Exploratory algorithms: held-out accuracy',fontsize=9,ha='left',va='top')
        for i,algorithm in enumerate(ALGORITHMS):
            r=summaries[summaries.algorithm==algorithm].iloc[0];d=folddata[folddata.algorithm==algorithm].sort_values('fold')
            color=ALG_COLORS[i]
            ax.errorbar(r.pooled_accuracy,i,xerr=np.array([[r.pooled_accuracy-r.ci_lower],[r.ci_upper-r.pooled_accuracy]]),
                        fmt='D',color=color,markersize=3.8,capsize=2,linewidth=1)
            bx.scatter(d.accuracy,i+np.linspace(-.14,.14,5),color='#7C8E99',s=13,zorder=2)
            bx.scatter(r.mean_fold_accuracy,i,color=color,marker='D',s=23,zorder=3)
        for axes in [ax,bx]:
            axes.set_ylim(10.6,-.8);axes.set_xlim(.58,1);axes.set_xticks([.6,.7,.8,.9,1]);axes.set_yticks(range(11));axes.tick_params(axis='y',length=0)
            axes.set_xlabel('Held-out accuracy');axes.spines['left'].set_visible(False);axes.tick_params(axis='x',labelsize=7)
        ax.set_yticklabels(ALGORITHMS,fontsize=7);bx.set_yticklabels([])
        ax.set_title('Pooled accuracy and 95% CI',fontsize=8);bx.set_title('Five folds and their mean',fontsize=8)
        fig.text(.155,.91,'A',fontsize=10,fontweight='bold');fig.text(.585,.91,'B',fontsize=10,fontweight='bold')
        handles=[Line2D([],[],marker='o',color='#7C8E99',linestyle='none',label='Individual outer fold',markersize=4),
                 Line2D([],[],marker='D',color='black',linestyle='none',label='Mean across outer folds',markersize=4)]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.745,.142),ncol=2,frameon=False,
                   fontsize=6.6,handletextpad=.35,columnspacing=1.1)
        fig.text(.055,.11,'Default thresholds: probability ≥0.5 for ten algorithms and SVM decision score ≥0. No threshold is optimized here.',fontsize=7)
        fig.text(.055,.065,'Pooled intervals use 2,000 recorded-group resamples conditional on fixed classifications. Fold points are dependent.',fontsize=7)
        fig.text(.055,.025,'Pooled accuracy and the mean fold accuracy differ because evaluation-fold sizes differ. Neither is apparent training accuracy.',fontsize=7)
        metric_file='SupplementaryFigure4_accuracy_pooled_metrics.csv';fold_file='SupplementaryFigure4_accuracy_fold_metrics.csv'
        self.source(summaries,metric_file);self.source(folddata,fold_file)
        self.save(fig,'SupplementaryFigure4C_Exploratory_accuracy',
                  'Supplementary Figure 4C. Held-out classification accuracy at default thresholds. '
                  'A, pooled accuracy with fixed-classification 95% intervals from 2,000 source-group resamples. '
                  'B, five individual outer-fold accuracies and their arithmetic mean. '
                  'Probability thresholds are 0.5 and the SVM decision-score threshold is 0. '
                  'These defaults differ from primary-model inner Youden thresholds. Fold sizes differ, so pooled and mean-fold accuracy can differ. '
                  'The folds share overlapping development data and are not independent replicates. Apparent training accuracy is not displayed.',
                  [metric_file,fold_file,predictions_file])
        self.numeric_checks.append({'figure':'S3B/S4C','algorithms':11,'fold_accuracy_rows':55,'classification_rows':len(classification),
                                    'threshold_optimization':False,'probability_threshold':.5,'SVM_score_threshold':0,'exact_threshold_ties':0})

    def s12_groups(self):
        data=self.read('patient_group_predictions.csv');metrics=self.read('patient_group_metrics.csv')
        fig,axes=plt.subplots(1,2,figsize=(180*MM,111*MM));fig.subplots_adjust(left=.09,right=.97,bottom=.27,top=.87,wspace=.38)
        fig.suptitle('Ranking source groups by their maximum selected-node score',x=.02,y=.99,ha='left',fontsize=9)
        coordinates=[]
        for model in MODELS:
            d=data[data.model==model];y=d.y_true_selected_nodes.to_numpy();score=d.max_node_score.to_numpy()
            assert len(d)==130 and int(y.sum())==45 and d.group_id.nunique()==130
            auc=metrics[(metrics.model==model)&(metrics.metric=='roc_auc')].iloc[0]
            ap=metrics[(metrics.model==model)&(metrics.metric=='average_precision')].iloc[0]
            assert np.isclose(roc_auc_score(y,score),auc.estimate,atol=1e-10) and np.isclose(average_precision_score(y,score),ap.estimate,atol=1e-10)
            fpr,tpr,threshold=roc_curve(y,score);precision,recall,cut=precision_recall_curve(y,score)
            axes[0].plot(fpr,tpr,color=COLORS[model],label=f'{LABEL[model]}: {auc.estimate:.3f} ({auc.ci_lower:.3f}-{auc.ci_upper:.3f})')
            axes[1].plot(recall,precision,color=COLORS[model],label=f'{LABEL[model]}: {ap.estimate:.3f} ({ap.ci_lower:.3f}-{ap.ci_upper:.3f})')
            for x,z,t in zip(fpr,tpr,threshold):coordinates.append({'model':model,'curve':'ROC','x':x,'y':z,'threshold':t if np.isfinite(t) else np.nan})
            for i,(x,z) in enumerate(zip(recall,precision)):coordinates.append({'model':model,'curve':'PR','x':x,'y':z,'threshold':cut[i] if i<len(cut) else np.nan})
        for i,ax in enumerate(axes):
            ax.set_xlim(0,1);ax.set_ylim(0,1.01);ax.legend(loc='upper left',bbox_to_anchor=(-.02,-.17),frameon=False,fontsize=6.8)
            ax.text(-.15,1.1,chr(65+i),transform=ax.transAxes,fontsize=10,fontweight='bold')
        axes[0].plot([0,1],[0,1],linestyle=':',color='#929292');axes[0].set_xlabel('1 - specificity');axes[0].set_ylabel('Sensitivity');axes[0].set_title('ROC-AUC',fontsize=8)
        axes[1].axhline(45/130,color='#929292',linestyle=':');axes[1].set_xlabel('Recall (sensitivity)');axes[1].set_ylabel('Precision');axes[1].set_title('Average precision',fontsize=8)
        fig.text(.05,.035,'130 groups: 45 contain at least one malignant selected node. Maximum scores are ranking scores, not calibrated patient risks.',fontsize=6.8)
        fig.text(.05,.012,'This outcome concerns selected nodes and does not establish whole-neck pathological stage. No group calibration or DCA is inferred.',fontsize=6.8)
        source='SupplementaryFigure12_group_ROC_PR_coordinates.csv';predfile='SupplementaryFigure12_selected_group_scores.csv'
        self.source(pd.DataFrame(coordinates),source);self.source(data,predfile)
        self.save(fig,'SupplementaryFigure12_Selected_group_ROC_PR',
                  'Supplementary Figure 12. Discrimination for recorded malignant status among selected nodes in each source group. '
                  'The analysis contains 130 source groups, including 45 with one or more malignant selected nodes. '
                  'Each model ranks groups using its maximum held-out selected-node probability. The maximum is not a calibrated patient probability. '
                  'A, ROC-AUC. B, average precision with a reference of 45/130. Legend intervals are the existing source-group bootstrap intervals. '
                  'This target does not establish whole-neck pN stage, and no group-level calibration or decision curve is implied.',[source,predfile])
        self.numeric_checks.append({'figure':'S12','groups':130,'groups_with_malignant_selected_nodes':45,'max_score_used_as_probability':False})

    def s13_reclassification(self):
        data=self.read('primary_predictions.csv');baseline=data[data.model=='Combined'].sort_values('node_id').reset_index(drop=True)
        groups=sorted(baseline.group_id.unique());mapping={g:i for i,g in enumerate(groups)}
        indexes=np.array([mapping[g] for g in baseline.group_id]);rng=np.random.default_rng(20260926)
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=self.bootstrap)[:,indexes]
        y=baseline.y_true.to_numpy();event_weight=weights*y;nonevent_weight=weights*(1-y)
        ne=event_weight.sum(axis=1);nn=nonevent_weight.sum(axis=1)
        assert (ne>0).all() and (nn>0).all()
        results=[];node_rows=[]
        for comparator in ['NodeRADS','Imaging']:
            old=data[data.model==comparator].sort_values('node_id').reset_index(drop=True)
            assert old.node_id.equals(baseline.node_id) and np.array_equal(old.y_true,y) and np.array_equal(old.group_id,baseline.group_id)
            delta=baseline.probability.to_numpy()-old.probability.to_numpy();direction=np.sign(delta)
            point_event=direction[y==1].mean();point_nonevent=-direction[y==0].mean()
            event_boot=(event_weight@direction)/ne;nonevent_boot=-(nonevent_weight@direction)/nn
            idi=delta[y==1].mean()-delta[y==0].mean();idi_boot=(event_weight@delta)/ne-(nonevent_weight@delta)/nn
            measures={'event_NRI':(point_event,event_boot),'nonevent_NRI':(point_nonevent,nonevent_boot),
                      'total_continuous_NRI':(point_event+point_nonevent,event_boot+nonevent_boot),'IDI':(idi,idi_boot)}
            for metric,(estimate,values) in measures.items():
                ci=np.quantile(values,[.025,.975])
                results.append({'comparison':'Combined - '+comparator,'metric':metric,'estimate':float(estimate),
                                'ci_lower':float(ci[0]),'ci_upper':float(ci[1]),'bootstrap_replicates':self.bootstrap,
                                'bootstrap_seed':20260926,'n_nodes':248,'malignant_nodes':84,'n_source_groups':130,
                                'analysis_role':'secondary_descriptive_no_confirmatory_P',
                                'interval_method':'fixed_oof_probabilities_paired_source_group_percentile_bootstrap'})
            for j,row in baseline.iterrows():
                node_rows.append({'comparison':'Combined - '+comparator,'node_id':row.node_id,'group_id':int(row.group_id),'fold':int(row.fold),
                                  'y_true':int(row.y_true),'combined_probability':float(row.probability),'comparator_probability':float(old.iloc[j].probability),
                                  'probability_change':float(delta[j]),'direction':int(direction[j])})
            self.numeric_checks.append({'figure':'S13','comparator':comparator,'event_up':int((direction[y==1]>0).sum()),
                                        'event_down':int((direction[y==1]<0).sum()),'nonevent_up':int((direction[y==0]>0).sum()),
                                        'nonevent_down':int((direction[y==0]<0).sum()),'ties':int((direction==0).sum()),
                                        'IDI':float(idi),'continuous_NRI':float(point_event+point_nonevent)})
        stats=pd.DataFrame(results);fig=plt.figure(figsize=(180*MM,103*MM));ax=fig.add_axes([.245,.28,.31,.57]);bx=fig.add_axes([.69,.28,.28,.57])
        fig.text(.02,.985,'Probability changes from the combined model: secondary descriptions',fontsize=9,ha='left',va='top')
        comparisons=['Combined - NodeRADS','Combined - Imaging']
        for i,comparison in enumerate(comparisons):
            d=stats[stats.comparison==comparison];r=d[d.metric=='IDI'].iloc[0]
            ax.errorbar(r.estimate,i,xerr=np.array([[r.estimate-r.ci_lower],[r.ci_upper-r.estimate]]),fmt='D',color=COLORS['Combined'],capsize=2,markersize=4)
            for j,(metric,color,marker) in enumerate([('event_NRI','#0072B2','o'),('nonevent_NRI','#D55E00','s'),('total_continuous_NRI','#009E73','D')]):
                r=d[d.metric==metric].iloc[0]
                bx.errorbar(r.estimate,i+(j-1)*.18,xerr=np.array([[r.estimate-r.ci_lower],[r.ci_upper-r.estimate]]),fmt=marker,color=color,capsize=2,markersize=4)
        for axes in [ax,bx]:
            axes.axvline(0,color='#666666',linestyle=':',linewidth=1);axes.set_ylim(1.55,-.7);axes.set_yticks([0,1]);axes.tick_params(axis='y',length=0);axes.spines['left'].set_visible(False)
        ax.set_yticklabels(['Combined vs\nNode-RADS LR','Combined vs\nImaging'],fontsize=7.5);bx.set_yticklabels([])
        idi_stats=stats[stats.metric=='IDI'];nri_stats=stats[stats.metric!='IDI']
        ax.set_xlim(min(-.03,idi_stats.ci_lower.min()-.03),idi_stats.ci_upper.max()+.04)
        bx.set_xlim(min(-.1,nri_stats.ci_lower.min()-.08),nri_stats.ci_upper.max()+.08)
        ax.set_title('IDI (95% CI)',fontsize=8);bx.set_title('Continuous NRI components (95% CI)',fontsize=8)
        ax.set_xlabel('Change in discrimination slope');bx.set_xlabel('Continuous NRI')
        fig.text(.195,.91,'A',fontsize=10,fontweight='bold');fig.text(.64,.91,'B',fontsize=10,fontweight='bold')
        legend=[Line2D([],[],color=c,marker=m,label=label) for c,m,label in [('#0072B2','o','Event'),('#D55E00','s','Non-event'),('#009E73','D','Total')]]
        bx.legend(handles=legend,loc='upper left',bbox_to_anchor=(-.12,-.17),ncol=3,frameon=False,fontsize=6.5,handlelength=1.2)
        fig.text(.055,.125,'248 nodes in 130 groups. CIs: 2,000 paired group resamples.',fontsize=7)
        fig.text(.055,.078,'NRI counts any probability increase/decrease. IDI changes the event/non-event mean probability separation.',fontsize=7)
        fig.text(.055,.025,'These calibration-sensitive descriptions use no clinical risk categories, no confirmatory P values and no clinical-benefit claim.',fontsize=7)
        source='SupplementaryFigure13_IDI_NRI_metrics.csv';changes='SupplementaryFigure13_node_probability_changes.csv'
        self.source(stats,source);self.source(pd.DataFrame(node_rows),changes)
        save_json(self.srcdir/'SupplementaryFigure13_methods.json',
                  {'event_NRI':'P(delta>0|y=1)-P(delta<0|y=1)','nonevent_NRI':'P(delta<0|y=0)-P(delta>0|y=0)',
                   'total_continuous_NRI':'event_NRI+nonevent_NRI','IDI':'mean(delta|y=1)-mean(delta|y=0)',
                   'delta':'combined_probability-comparator_probability','ties':'exact equality contributes 0 to directional NRI',
                   'clinical_risk_categories':None,'clinical_benefit_claim':False,'new_confirmatory_tests':False,
                   'sources':[{'title':'Extensions of net reclassification improvement calculations to measure usefulness of new biomarkers',
                               'PMID':'21204120','DOI':'10.1002/sim.4085','url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC3341973/'},
                              {'title':'Discrimination slope and integrated discrimination improvement - properties, relationships and impact of calibration',
                               'PMID':'27699818','DOI':'10.1002/sim.7139','url':'https://pubmed.ncbi.nlm.nih.gov/27699818/'}]})
        self.save(fig,'SupplementaryFigure13_IDI_NRI',
                  'Supplementary Figure 13. Secondary descriptive IDI and continuous NRI for the combined model against each fitted comparator. '
                  'A, IDI is the combined-minus-comparator change in the difference between mean event and non-event probabilities. '
                  'B, event NRI is the fraction of malignant nodes with an increased probability minus the fraction with a decrease. '
                  'Non-event NRI reverses this direction for benign nodes. Total NRI is their sum. Exact ties contribute zero. '
                  'Intervals are 95% percentile intervals from 2,000 paired source-group resamples conditional on fixed out-of-fold probabilities. '
                  'These measures are sensitive to probability calibration, use no clinically established categories, and do not demonstrate changes in clinical decisions or outcomes. '
                  'They are secondary descriptions without confirmatory P values.',[source,changes,'SupplementaryFigure13_methods.json'])

    def s16_raw_node_rads(self):
        cohort=self.read('cohort_primary.csv').sort_values('node_id').reset_index(drop=True)
        pred=self.read('primary_predictions.csv');raw=self.read('raw_noderads_benchmark.csv');metrics=self.read('metrics_primary.csv')
        assert len(cohort)==248 and cohort.group_id.nunique()==130 and int(cohort.outcome.sum())==84
        assert cohort.node_rads.notna().all() and set(cohort.node_rads.unique())=={1,2,3,4,5}
        y=cohort.outcome.to_numpy();grade=cohort.node_rads.to_numpy();records=[];coordinates=[]
        fig,axes=plt.subplots(1,2,figsize=(180*MM,124*MM));fig.subplots_adjust(left=.09,right=.975,bottom=.34,top=.87,wspace=.39)
        fig.suptitle('Original Node-RADS grades and fitted held-out predictions',x=.02,y=.99,ha='left',fontsize=9)
        raw_metric=raw[(raw.benchmark=='RawNodeRADS')&(raw.metric=='roc_auc')].iloc[0]
        assert np.isclose(roc_auc_score(y,grade),raw_metric.estimate,atol=1e-10)
        score_sets=[('Raw ordinal Node-RADS',grade,raw_metric,'#333333',':')]
        for model in ['NodeRADS','Combined']:
            d=pred[pred.model==model].sort_values('node_id').reset_index(drop=True)
            assert d.node_id.equals(cohort.node_id) and np.array_equal(d.y_true,y) and np.array_equal(d.group_id,cohort.group_id)
            row=metrics[(metrics.model==model)&(metrics.metric=='roc_auc')].iloc[0]
            assert np.isclose(roc_auc_score(y,d.probability),row.estimate,atol=1e-10)
            score_sets.append((LABEL[model],d.probability.to_numpy(),row,COLORS[model],'--' if model=='NodeRADS' else '-'))
            if model=='NodeRADS':
                for fold in range(1,6):
                    mask=cohort.outer_fold.eq(fold).to_numpy()
                    assert np.isclose(roc_auc_score(y[mask],grade[mask]),roc_auc_score(y[mask],d.probability.to_numpy()[mask]),atol=1e-12)
        for name,score,row,color,style in score_sets:
            fpr,tpr,threshold=roc_curve(y,score)
            axes[0].plot(fpr,tpr,color=color,linestyle=style,linewidth=1.3,
                         label=f'{name}: {row.estimate:.3f} ({row.ci_lower:.3f}-{row.ci_upper:.3f})')
            records.append({'score':name,'roc_auc':float(row.estimate),'ci_lower':float(row.ci_lower),'ci_upper':float(row.ci_upper),
                            'n_nodes':248,'malignant_nodes':84,'n_source_groups':130,
                            'fitted_score':name!='Raw ordinal Node-RADS',
                            'interval_source':'raw_noderads_benchmark.csv' if name=='Raw ordinal Node-RADS' else 'metrics_primary.csv'})
            for x,z,t in zip(fpr,tpr,threshold):coordinates.append({'score':name,'false_positive_rate':x,'sensitivity':z,'threshold':t if np.isfinite(t) else np.nan})
        axes[0].plot([0,1],[0,1],color='#999999',linestyle=':',linewidth=.9)
        axes[0].set_xlim(0,1);axes[0].set_ylim(0,1.01);axes[0].set_xlabel('1 - specificity');axes[0].set_ylabel('Sensitivity')
        axes[0].set_title('ROC-AUC in the same selected nodes',fontsize=8)
        axes[0].legend(loc='upper left',bbox_to_anchor=(-.02,-.16),frameon=False,fontsize=6.5,handlelength=2.0,handletextpad=.4)
        fractions=[]
        for g in range(1,6):
            d=cohort[cohort.node_rads==g];total=len(d);malignant=int(d.outcome.sum())
            fractions.append({'grade':g,'malignant_nodes':malignant,'benign_nodes':total-malignant,'total_nodes':total,
                              'recorded_malignant_fraction':malignant/total,'source_groups':d.group_id.nunique(),
                              'summary_scope':'selected_cohort_descriptive_not_population_risk'})
        assert [(v['malignant_nodes'],v['total_nodes']) for v in fractions]==[(0,85),(22,90),(37,48),(21,21),(4,4)]
        values=[v['recorded_malignant_fraction'] for v in fractions]
        axes[1].bar(range(1,6),values,color='#CBD0D4',edgecolor='#333333',linewidth=.6,width=.64)
        for v in fractions:
            axes[1].text(v['grade'],v['recorded_malignant_fraction']+.04,
                         f'{v["malignant_nodes"]}/{v["total_nodes"]}',ha='center',va='bottom',fontsize=7.5)
        axes[1].set_xlim(.4,5.6);axes[1].set_ylim(0,1.16);axes[1].set_xticks(range(1,6));axes[1].set_yticks(np.arange(0,1.01,.2))
        axes[1].set_xlabel('Original Node-RADS grade');axes[1].set_ylabel('Recorded malignant fraction')
        axes[1].set_title('Malignant / total selected nodes',fontsize=8)
        for i,ax in enumerate(axes):ax.text(-.15,1.12,chr(65+i),transform=ax.transAxes,fontsize=10,fontweight='bold')
        fig.text(.055,.125,'Raw ordinal scores retain ties. Pooled fold-specific logistic mappings can separate equal grades across folds.',fontsize=7)
        fig.text(.055,.08,'Within each outer fold, the raw grade and fitted Node-RADS LR have identical ROC-AUC.',fontsize=7)
        fig.text(.055,.035,'Grade fractions are descriptive in this selected cohort. Grade 5 has four nodes; observed 0% or 100% does not establish population risk.',fontsize=6.8)
        metric_file='SupplementaryFigure16_raw_NodeRADS_ROC_metrics.csv';coord_file='SupplementaryFigure16_raw_NodeRADS_ROC_coordinates.csv'
        grade_file='SupplementaryFigure16_NodeRADS_grade_counts.csv'
        self.source(pd.DataFrame(records),metric_file);self.source(pd.DataFrame(coordinates),coord_file);self.source(pd.DataFrame(fractions),grade_file)
        self.numeric_checks.append({'figure':'S16','same_nodes':248,'source_groups':130,'raw_AUC':float(raw_metric.estimate),
                                    'within_each_fold_raw_and_LR_AUC_identical':True,
                                    'grade_malignant_total':[[v['malignant_nodes'],v['total_nodes']] for v in fractions],
                                    'independent_node_binomial_CI_added':False})
        self.save(fig,'SupplementaryFigure16_Raw_NodeRADS_ROC_grades',
                  'Supplementary Figure 16. Original Node-RADS grades and fitted held-out node predictions. '
                  'A, ROC curves for the unfitted ordinal 1 to 5 score, the Node-RADS ridge logistic model and the combined model on the same 248 selected nodes in 130 source groups. '
                  'Legend confidence intervals are the existing fixed-score source-group bootstrap intervals. Raw scores retain tied grades. '
                  'Pooling probabilities from fold-specific logistic mappings can separate nodes with the same grade across folds. '
                  'Within each outer fold the raw ordinal and fitted Node-RADS logistic score have identical ROC-AUC. '
                  'B, recorded malignant fractions by grade, labeled with exact malignant/total counts. '
                  'No independent-node binomial intervals are imposed. Grades 4 and 5 contain 21 and 4 nodes, respectively. '
                  'The observed fractions, including 0% and 100%, describe this selected cohort and do not establish population risks or external validation.',
                  [metric_file,coord_file,grade_file])

    def append_raw_node_rads(self):
        """Append just S16, preserving all existing figure assets and QA records."""
        setup_style();self.s16_raw_node_rads()
        qa_path=self.figdir/'extended_validation_figures_QA.json'
        prior=json.loads(qa_path.read_text(encoding='utf-8'))
        stem=self.manifest[0]['figure']
        prior['figures']=[v for v in prior['figures'] if v['figure']!=stem]+self.manifest
        prior['numeric_checks']=[v for v in prior['numeric_checks'] if v.get('figure')!='S16']+self.numeric_checks
        for name,checksum in self.input_hashes.items():
            if name in prior['input_sha256']:assert prior['input_sha256'][name]==checksum
        prior['input_sha256'].update(self.input_hashes)
        prior['visual_review_status']='Existing figures passed; appended S16 awaits individual PNG review.'
        save_json(qa_path,prior)
        cap_path=self.figdir/'extended_validation_figure_captions.json'
        caps=json.loads(cap_path.read_text(encoding='utf-8'))
        caption=self.manifest[0]['caption']
        caps=[v for v in caps if not v['path'].endswith(stem+'.png')]
        caps.append({'title':caption.split('. ',1)[0]+'. '+caption.split('. ',2)[1],
                     'text':caption.split('. ',2)[2],'path':str(self.figdir/(stem+'.png')),'supplementary':True})
        save_json(cap_path,caps)
        print('Appended S16 only. Existing figure files and names were preserved.',flush=True)

    def s17_paired_auc(self):
        cohort=self.read('cohort_primary.csv');data=self.read('primary_predictions.csv')
        metrics=self.read('metrics_primary.csv');comparisons=self.read('comparisons_primary.csv')
        node_ids=cohort.node_id.tolist()
        aligned={m:data[data.model==m].set_index('node_id').loc[node_ids] for m in MODELS}
        y=aligned[MODELS[0]].y_true.to_numpy();group_ids=aligned[MODELS[0]].group_id.astype(str).to_numpy()
        groups=sorted(set(group_ids));lookup={g:i for i,g in enumerate(groups)}
        indexes=np.array([lookup[g] for g in group_ids]);rng=np.random.default_rng(20260926)
        # Match the original evaluation's lexical source-group order and integer draws exactly.
        draws=np.stack([np.bincount(rng.integers(0,len(groups),size=len(groups)),minlength=len(groups))
                        for _ in range(self.bootstrap)])
        weights=draws[:,indexes].astype(float);auc={};boot={}
        for m in MODELS:
            d=aligned[m]
            assert np.array_equal(d.y_true,y) and np.array_equal(d.group_id.astype(str),group_ids)
            auc[m]=float(roc_auc_score(y,d.probability))
            expected=metrics[(metrics.model==m)&(metrics.metric=='roc_auc')].iloc[0]
            assert np.isclose(auc[m],expected.estimate,atol=1e-10)
            boot[m],_=clustered_score_intervals(y,d.probability.to_numpy(),weights)
            assert np.allclose(np.quantile(boot[m],[.025,.975]),[expected.ci_lower,expected.ci_upper],atol=1e-10)
        pairs={};rows=[]
        for old in ['NodeRADS','Imaging']:
            existing=comparisons[(comparisons.metric=='roc_auc')&(comparisons.comparison=='Combined - '+old)].iloc[0]
            calculated=np.quantile(boot['Combined']-boot[old],[.025,.975])
            assert np.allclose(calculated,[existing.ci_lower,existing.ci_upper],atol=1e-10)
            pairs[('Combined',old)]=(float(existing.difference),float(existing.ci_lower),float(existing.ci_upper),'primary_existing_two_comparison_family')
        delta=auc['NodeRADS']-auc['Imaging'];lo,hi=np.quantile(boot['NodeRADS']-boot['Imaging'],[.025,.975])
        pairs[('NodeRADS','Imaging')]=(float(delta),float(lo),float(hi),'secondary_descriptive_no_P')
        difference=np.zeros((3,3));labels=[]
        for row,m1 in enumerate(MODELS):
            cell=[]
            for col,m2 in enumerate(MODELS):
                if m1==m2:estimate,lower,upper,role=0.,0.,0.,'same_model_no_comparison'
                elif (m1,m2) in pairs:estimate,lower,upper,role=pairs[(m1,m2)]
                else:
                    reverse,low,high,role=pairs[(m2,m1)];estimate,lower,upper=-reverse,-high,-low
                difference[row,col]=estimate
                cell.append('0.000' if row==col else f'{estimate:+.3f}\n({lower:+.3f} to {upper:+.3f})')
                rows.append({'row_model':m1,'column_model':m2,'auc_difference_row_minus_column':estimate,
                             'ci_lower':lower,'ci_upper':upper,'analysis_role':role,
                             'n_nodes':248,'malignant_nodes':84,'n_source_groups':130,
                             'bootstrap_replicates':self.bootstrap,'bootstrap_seed':20260926,
                             'sampling':'original_evaluate_results_lexical_groups_rng_integer_draws',
                             'inference':'paired_source_group_bootstrap_fixed_oof_predictions_no_new_P_values'})
            labels.append(cell)
        assert np.allclose(difference,-difference.T,atol=1e-12)
        fig=plt.figure(figsize=(180*MM,140*MM));ax=fig.add_axes([.18,.28,.70,.60])
        fig.text(.02,.985,'Paired ROC-AUC differences in the same held-out nodes',fontsize=9,ha='left',va='top')
        limit=float(np.abs(difference).max())
        cmap=LinearSegmentedColormap.from_list('paired_difference',['#BDD5E5','white','#E5C0C8'])
        im=ax.imshow(difference,vmin=-limit,vmax=limit,cmap=cmap,interpolation='nearest',aspect='equal')
        for row in range(3):
            for col in range(3):ax.text(col,row,labels[row][col],ha='center',va='center',fontsize=7.5,linespacing=1.6)
        display=['Node-RADS LR','Imaging parameters','Combined']
        ax.set_xticks(range(3),display);ax.set_yticks(range(3),display);ax.tick_params(length=0,labelsize=7.5)
        ax.set_xlabel('Column model',labelpad=8);ax.set_ylabel('Row model',labelpad=8)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        colorax=fig.add_axes([.90,.33,.018,.49]);cb=fig.colorbar(im,cax=colorax)
        cb.set_label('AUC difference: row minus column',fontsize=7.5);cb.ax.tick_params(labelsize=7)
        fig.text(.055,.18,'Cells show paired AUC differences (95% CIs). n = 248 nodes in 130 source-recorded groups. Diagonal cells are zero.',fontsize=7)
        fig.text(.055,.115,'CIs use the original 2,000 group-bootstrap draws and fixed held-out predictions; reverse cells negate both estimate and CI.',fontsize=7)
        fig.text(.055,.055,'The two combined-model comparisons retain their primary intervals. Node-RADS LR minus imaging is secondary and descriptive.',fontsize=6.8)
        source='SupplementaryFigure17_paired_AUC_differences.csv';self.source(pd.DataFrame(rows),source)
        self.numeric_checks.append({'figure':'S17','all_three_primary_AUCs_reproduced':True,
                                    'primary_AUC_CIs_reproduced_same_draws':True,
                                    'existing_combined_comparison_intervals_reused_exactly':True,
                                    'secondary_NodeLR_minus_Imaging_difference':float(delta),
                                    'secondary_NodeLR_minus_Imaging_CI':[float(lo),float(hi)],
                                    'new_P_values':False,'DeLong_independent_node_assumption':False})
        self.save(fig,'SupplementaryFigure17_Paired_AUC_difference_heatmap',
                  'Supplementary Figure 17. Paired ROC-AUC differences between the three primary models. '
                  'Each cell is the row model minus the column model on the same 248 held-out nodes in 130 source-recorded groups. '
                  'Cells contain differences and 95% paired source-group percentile bootstrap intervals conditional on the fixed out-of-fold predictions. '
                  'The original 2,000 bootstrap draws and group ordering are reproduced. The two combined-model comparisons reuse the exact intervals already reported in Table 3. '
                  'The Node-RADS logistic versus imaging comparison is secondary and descriptive. No additional P values are computed, '
                  'and the original two-comparison Holm family is unchanged. Diagonal cells compare a model with itself. '
                  'Reverse cells negate the estimate and exchange and negate the confidence limits. '
                  'This display replaces an independent-node significance comparison and does not constitute a new train/test split or external validation.',[source])

    def append_paired_auc(self):
        setup_style();self.s17_paired_auc()
        qa_path=self.figdir/'extended_validation_figures_QA.json';prior=json.loads(qa_path.read_text(encoding='utf-8'))
        stem=self.manifest[0]['figure']
        prior['figures']=[v for v in prior['figures'] if v['figure']!=stem]+self.manifest
        prior['numeric_checks']=[v for v in prior['numeric_checks'] if v.get('figure')!='S17']+self.numeric_checks
        for name,checksum in self.input_hashes.items():
            if name in prior['input_sha256']:assert prior['input_sha256'][name]==checksum
        prior['input_sha256'].update(self.input_hashes)
        prior['visual_review_status']='Existing figures passed; appended S17 awaits individual PNG review.'
        save_json(qa_path,prior)
        cap_path=self.figdir/'extended_validation_figure_captions.json';caps=json.loads(cap_path.read_text(encoding='utf-8'))
        caption=self.manifest[0]['caption'];caps=[v for v in caps if not v['path'].endswith(stem+'.png')]
        caps.append({'title':caption.split('. ',1)[0]+'. '+caption.split('. ',2)[1],
                     'text':caption.split('. ',2)[2],'path':str(self.figdir/(stem+'.png')),'supplementary':True})
        save_json(cap_path,caps)
        print('Appended S17 only. Existing figure files and primary comparison results were preserved.',flush=True)

    def run(self):
        setup_style();self.s3_confusion();self.s4_algorithms();self.s5_sensitivity()
        self.s3b_s4c_default_classification();self.s12_groups();self.s13_reclassification()
        result={'generated_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'input_sha256':self.input_hashes,'model_refitting':False,'algorithm_reselection':False,
                'numeric_checks':self.numeric_checks,'figures':self.manifest,
                'visual_review_status':'Awaiting individual PNG inspection'}
        save_json(self.figdir/'extended_validation_figures_QA.json',result)
        save_json(self.figdir/'extended_validation_figure_captions.json',
                  [{'title':v['caption'].split('. ',1)[0]+'. '+v['caption'].split('. ',2)[1],
                    'text':v['caption'].split('. ',2)[2],'path':str(self.figdir/(v['figure']+'.png')),'supplementary':True}
                   for v in self.manifest])
        print(f'Complete: {len(self.manifest)} extended validation figures, {len(self.manifest)*4} exported image files. No model fitted.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--add-raw-noderads',action='store_true')
    parser.add_argument('--add-paired-auc',action='store_true')
    arguments=parser.parse_args();instance=ExtendedFigures(arguments.root,arguments.bootstrap)
    if arguments.add_raw_noderads and arguments.add_paired_auc:parser.error('Choose one append operation per run.')
    if arguments.add_paired_auc:instance.append_paired_auc()
    elif arguments.add_raw_noderads:instance.append_raw_node_rads()
    else:instance.run()
