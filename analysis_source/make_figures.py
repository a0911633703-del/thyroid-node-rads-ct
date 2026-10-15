"""Make publication figures from frozen analysis outputs, without fitting models.

Default Python: the recorded Python environment
Run only after the main and evaluation scripts finish writing their inputs.
Example: python make_figures.py --figures 1 3 4 5 S1

All figures use matplotlib Agg, exact 180 mm canvas width, Arial, editable
PDF/SVG text, and 300 dpi PNG/TIFF. Confidence intervals are read from the
evaluation output. They are never replaced with independent-node intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['NodeRADS', 'Imaging', 'Combined']
COLORS = {'NodeRADS': '#0072B2', 'Imaging': '#D55E00', 'Combined': '#009E73'}
LABELS = {'NodeRADS': 'Node-RADS LR', 'Imaging': 'Imaging parameters', 'Combined': 'Combined'}
STYLES = {'NodeRADS': '--', 'Imaging': '-.', 'Combined': '-'}
MARKERS = {'NodeRADS': 'o', 'Imaging': 's', 'Combined': '^'}
MM = 1 / 25.4
PRETTY = {
 'node_rads': 'Node-RADS', 'NodeRADS': 'Node-RADS', 'noderads': 'Node-RADS',
 'node_rads_score': 'Node-RADS', 'longAxis': 'Long-axis diameter', 'long_axis_mm': 'Long-axis diameter',
 'shortAxis': 'Short-axis diameter', 'short_axis_mm': 'Short-axis diameter',
 'ct_70kev': '70-keV attenuation', 'ct_100kev': '100-keV attenuation',
 'nc_ct': 'Unenhanced attenuation', 'ce_ct': 'Enhanced attenuation',
 'nc_ed': 'Unenhanced electron density', 'ce_ed': 'Enhanced electron density',
 'nc_zeff': 'Unenhanced Zeff', 'ce_zeff': 'Enhanced Zeff',
 'iodine': 'Iodine concentration', 'iodine_mg_ml': 'Iodine concentration',
 'ce_iodine': 'Enhanced iodine concentration', 'nc_iodine': 'Unenhanced iodine concentration', 'ic': 'Iodine concentration',
 'ct_ratio_artery': 'Node / artery CT ratio', 'ct_ratio_muscle': 'Node / muscle CT ratio',
 'casupp_30': 'Ca-suppression setting 30', 'casupp_70': 'Ca-suppression setting 70',
 'ca_supp_30': 'Ca-suppression setting 30', 'ca_supp_70': 'Ca-suppression setting 70',
 'ce_ct_artery': 'Arterial reference attenuation', 'ce_ct_muscle': 'Muscle reference attenuation',
 'spectral_slope': 'Spectral slope', 'delta_ct': 'Enhancement increment',
}


def setup_style() -> None:
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'axes.titleweight': 'normal',
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black',
        'ytick.color': 'black', 'axes.edgecolor': 'black', 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'lines.linewidth': 1.35, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 300,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'aizhaodong_20260926', 'axes.unicode_minus': False,
    })


class Plotter:
    def __init__(self, root: Path):
        self.root = root
        self.analysis = root / 'analysis'
        self.figdir = root / 'figures_tables' / 'figures'
        self.srcdir = root / 'figures_tables' / 'source_data'
        self.figdir.mkdir(parents=True, exist_ok=True)
        self.srcdir.mkdir(parents=True, exist_ok=True)
        self.manifest = []
        self.inputs = {}
        self.predictions = None
        self.metrics = None

    def read_csv(self, name: str, columns: list[str]) -> pd.DataFrame:
        path = self.analysis / name
        if not path.exists():
            raise FileNotFoundError(f'Required analysis file not ready: {path}')
        self.inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f'{name}: missing columns {missing}')
        return frame

    def source(self, frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(self.srcdir / name, index=False, encoding='utf-8-sig')

    def save(self, fig, stem: str, caption: str, source_files: list[str]) -> None:
        # Do not use bbox_inches=tight: preserve the requested 180 mm canvas.
        fig.canvas.draw()
        files = []
        for ext in ['pdf', 'svg', 'png', 'tif']:
            path = self.figdir / f'{stem}.{ext}'
            kwargs = {'dpi': 300, 'facecolor': 'white'}
            if ext == 'tif':
                kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
            fig.savefig(path, **kwargs)
            files.append(path.name)
        width_mm, height_mm = np.asarray(fig.get_size_inches()) * 25.4
        if not np.isclose(width_mm, 180):
            raise AssertionError('Unexpected figure width')
        self.manifest.append({'figure': stem, 'files': files, 'width_mm': float(width_mm),
                              'height_mm': float(height_mm), 'dpi_raster': 300,
                              'source_files': source_files, 'caption': caption})
        plt.close(fig)
        logging.info('Created %s', stem)

    def load_predictions(self) -> pd.DataFrame:
        if self.predictions is not None:
            return self.predictions
        d = self.read_csv('primary_predictions.csv', ['node_id','group_id','fold','y_true','model','probability'])
        d = d[d.model.isin(MODELS)].copy()
        if set(d.model) != set(MODELS):
            raise ValueError('Primary predictions must contain all three prespecified models')
        if d.duplicated(['node_id','model']).any():
            raise ValueError('Expected exactly one held-out prediction per node and model')
        if d[['node_id','group_id','fold','y_true','probability']].isna().any().any():
            raise ValueError('Missing identifying data, outcomes or predictions')
        if not d.y_true.isin([0,1]).all() or not d.probability.between(0,1).all():
            raise ValueError('Invalid binary outcome or probability')
        reference = d[d.model==MODELS[0]].set_index('node_id').sort_index()
        if reference.groupby('group_id').fold.nunique().max() != 1:
            raise ValueError('A source-record group appears in multiple outer folds')
        for model in MODELS[1:]:
            cur = d[d.model==model].set_index('node_id').sort_index()
            if not reference[['group_id','fold','y_true']].equals(cur[['group_id','fold','y_true']]):
                raise ValueError('Model predictions do not refer to the same nodes/outcomes/folds')
        self.predictions = d
        return d

    def metric(self, model: str, metric: str) -> pd.Series:
        if self.metrics is None:
            self.metrics = self.read_csv('metrics_primary.csv', ['model','metric','estimate','ci_lower','ci_upper'])
        a = self.metrics[(self.metrics.model==model)&(self.metrics.metric==metric)]
        if len(a) != 1:
            raise ValueError(f'Expected one metric row: {model} / {metric}')
        return a.iloc[0]

    def metric_label(self, model: str, metric: str, label: str) -> str:
        r = self.metric(model, metric)
        name = 'Imaging' if model == 'Imaging' else LABELS[model]
        if pd.notna(r.ci_lower) and pd.notna(r.ci_upper):
            return f'{name}: {label} {r.estimate:.3f} ({r.ci_lower:.3f}-{r.ci_upper:.3f})'
        return f'{name}: {label} {r.estimate:.3f} (CI unavailable)'

    @staticmethod
    def letter(ax, text: str, x=-.15, y=1.06):
        ax.text(x, y, text, fontsize=10, fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)

    def figure1(self) -> None:
        fig = plt.figure(figsize=(180*MM, 123*MM))
        ax = fig.add_axes([.025,.02,.95,.95]); ax.set_axis_off()
        def box(x,y,w,h,text,fill='#F6F7F8',edge='#AAB0B5',fs=8,lw=.6):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.008',
                                      linewidth=lw,edgecolor=edge,facecolor=fill))
            ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.4)
        def arrow(x1,y1,x2,y2):
            ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,
                                        linewidth=.8,color='#555555'))
        ax.text(.01,.975,'A',fontweight='bold',fontsize=10)
        ax.text(.06,.975,'Data included in this reanalysis',fontsize=9)
        box(.035,.770,.37,.137,'Source workbook\n254 nodes / 131 recorded patient groups\n88 malignant / 166 benign',fs=7.6)
        box(.590,.770,.375,.137,'Primary analysis: short axis >3 mm\n248 nodes / 130 recorded patient groups\n84 malignant / 164 benign',fill='#EEF7F3',edge=COLORS['Combined'],fs=7.6)
        arrow(.413,.838,.578,.838)
        box(.41,.665,.18,.090,'Excluded: 6 nodes\nShort axis ≤3 mm',fs=7.5)
        arrow(.495,.824,.495,.760)
        ax.text(.05,.635,'Source-recorded patient groups are retained; unique hospital patient identities remain unverified.',fontsize=7.4)
        ax.text(.01,.585,'B',fontweight='bold',fontsize=10)
        ax.text(.06,.585,'Shared outer-fold allocation and grouped nested validation',fontsize=9)
        box(.035,.412,.93,.12,'Five outer folds assigned once to all 254 source nodes\nAll nodes from one recorded patient group stay together\nThe same mapping is retained for primary and sensitivity analyses',fill='#F5F8FA',fs=7.7)
        # Five fold tiles denote train/evaluation roles, not observed counts.
        tile_y=.358; width=.145; start=.137
        for k in range(5):
            color=COLORS['Combined'] if k==4 else '#CBDDE8'
            ax.add_patch(Rectangle((start+k*width,tile_y),width-.009,.033,color=color,lw=0))
            ax.text(start+k*width+(width-.009)/2,tile_y+.0165,f'Fold {k+1}',ha='center',va='center',fontsize=7.2,
                    color='white' if k==4 else 'black')
        box(.035,.14,.545,.15,'Outer training groups\nThree inner folds select regularization\nImputation, scaling and threshold selection\nuse training data only',fs=7.7)
        box(.645,.14,.32,.15,'Held-out outer groups\nFreeze preprocessing and model\nPredict held-out nodes',fill='#EEF7F3',edge=COLORS['Combined'],fs=7.7)
        arrow(.35,.35,.30,.298); arrow(.79,.35,.80,.298)
        arrow(.588,.215,.635,.215)
        ax.text(.05,.082,'Repeat the outer evaluation for all five folds; pool held-out predictions.',fontsize=7.7)
        ax.text(.05,.034,'Primary comparisons: Node-RADS, imaging parameters, and their feature-level combination.',fontsize=7.7)
        source=pd.DataFrame([
            {'stage':'source_workbook','n_nodes':254,'n_source_groups':131,'malignant_nodes':88,'benign_nodes':166,'criterion':'all cleaned source records'},
            {'stage':'excluded','n_nodes':6,'n_source_groups':np.nan,'malignant_nodes':4,'benign_nodes':2,'criterion':'short-axis diameter <=3 mm'},
            {'stage':'primary','n_nodes':248,'n_source_groups':130,'malignant_nodes':84,'benign_nodes':164,'criterion':'short-axis diameter >3 mm'},
        ])
        self.source(source,'Figure1_cohort_counts.csv')
        sources=['Figure1_cohort_counts.csv']
        foldpath=self.analysis/'outer_fold_assignment.csv'
        if foldpath.exists():
            mapping=self.read_csv('outer_fold_assignment.csv',['node_id','group_id','outer_fold'])
            if len(mapping)!=254:raise ValueError('Shared outer-fold map must contain all 254 source nodes')
            self.source(mapping,'Figure1_shared_outer_fold_mapping.csv')
            sources.append('Figure1_shared_outer_fold_mapping.csv')
        caption=('Figure 1. Available source records and grouped nested internal validation. The flow starts with the supplied source workbook, '
                 'not an unverified earlier clinical screening population. Group counts refer to source-recorded patient identifiers, whose correspondence '
                 'to unique hospital patients has not been independently verified. Six nodes with short-axis diameter ≤3 mm are omitted from '
                 'the primary analysis. The five-fold outer mapping is first assigned once on all 254 source nodes and retained after short-axis filtering '
                 'and across sensitivity analyses. Three inner folds are used for tuning within each outer training set. '
                 'The illustrated held-out fold rotates across all five folds.')
        self.save(fig,'Figure1_Cohort_and_nested_validation',caption,sources)

    def figure3(self) -> None:
        data = self.load_predictions()
        fig, axes = plt.subplots(1,2,figsize=(180*MM,112*MM))
        fig.subplots_adjust(left=.087,right=.98,top=.865,bottom=.295,wspace=.28)
        ax,bx=axes; self.letter(ax,'A');self.letter(bx,'B')
        ax.set_title('Receiver operating characteristic',pad=10)
        bx.set_title('Precision-recall',pad=10)
        out=[]
        for model in MODELS:
            g=data[data.model==model]
            fpr,tpr,thresholds=roc_curve(g.y_true,g.probability,drop_intermediate=False)
            precision,recall,pr_threshold=precision_recall_curve(g.y_true,g.probability)
            # Confirm that the source metrics and plotted predictions agree.
            for metric,val in [('roc_auc',roc_auc_score(g.y_true,g.probability)),('average_precision',average_precision_score(g.y_true,g.probability))]:
                if not np.isclose(float(self.metric(model,metric).estimate),val,atol=5e-6):
                    raise ValueError(f'Metric mismatch for {model} {metric}')
            ax.plot(fpr,tpr,color=COLORS[model],ls=STYLES[model],label=self.metric_label(model,'roc_auc','AUC'))
            bx.step(recall,precision,where='post',color=COLORS[model],ls=STYLES[model],
                    label=self.metric_label(model,'average_precision','AP'))
            out.extend({'model':model,'curve':'ROC','x':x,'y':y,'threshold':t} for x,y,t in zip(fpr,tpr,thresholds))
            out.extend({'model':model,'curve':'PR','x':x,'y':y,'threshold':t} for x,y,t in zip(recall,precision,np.r_[pr_threshold,np.nan]))
        prevalence=float(data[data.model==MODELS[0]].y_true.mean())
        ax.plot([0,1],[0,1],color='#777777',ls=':',lw=.85,zorder=0)
        bx.axhline(prevalence,color='#777777',ls=':',lw=.85,zorder=0)
        bx.text(.02,prevalence+.025,f'Sample prevalence = {prevalence:.3f}',fontsize=7,color='#444444')
        for a in axes:
            a.set_xlim(0,1);a.set_ylim(0,1.01)
            a.set_xticks(np.arange(0,1.01,.2));a.set_yticks(np.arange(0,1.01,.2))
            a.set_box_aspect(1)
            a.legend(loc='upper left',bbox_to_anchor=(-.03,-.20),frameon=False,handlelength=2.7,
                     borderaxespad=0,labelspacing=.6,fontsize=7.3)
        ax.set_xlabel('1 - specificity');ax.set_ylabel('Sensitivity')
        bx.set_xlabel('Recall (sensitivity)');bx.set_ylabel('Precision (positive predictive value)')
        n=len(data[data.model==MODELS[0]])
        fig.text(.087,.958,f'Held-out predictions: {n} nodes in {data[data.model==MODELS[0]].group_id.nunique()} recorded patient groups',fontsize=8)
        fig.text(.087,.025,'AP = average precision. 95% CIs use recorded patient group bootstrap of fixed held-out predictions.',fontsize=7.2)
        self.source(pd.DataFrame(out),'Figure3_ROC_PR_coordinates.csv')
        self.source(data,'Figure3_heldout_predictions.csv')
        self.source(self.metrics[self.metrics.metric.isin(['roc_auc','average_precision'])], 'Figure3_AUC_AP_intervals.csv')
        self.save(fig,'Figure3_ROC_and_precision_recall',
                  'Figure 3. Node-level discrimination from pooled outer-fold held-out predictions. A, receiver operating characteristic curves. '
                  'B, precision-recall curves. Average precision (AP) is reported rather than trapezoidal PR area. Confidence intervals are imported '
                  'from paired recorded patient group bootstrap evaluation conditional on the fixed held-out predictions. The horizontal PR reference '
                  'is the observed malignant-node proportion in the selected analysis sample.',
                  ['Figure3_ROC_PR_coordinates.csv','Figure3_heldout_predictions.csv','Figure3_AUC_AP_intervals.csv'])

    def figure4(self) -> None:
        data=self.load_predictions()
        cal=self.read_csv('calibration_bins.csv',['model','bin','mean_predicted_probability','observed_event_fraction','n_nodes','events','observed_ci_lower','observed_ci_upper'])
        dca=self.read_csv('dca_source.csv',['model','threshold_probability','net_benefit','ci_lower','ci_upper'])
        fig,axes=plt.subplots(1,2,figsize=(180*MM,112*MM))
        fig.subplots_adjust(left=.087,right=.98,top=.865,bottom=.295,wspace=.31)
        ax,bx=axes;self.letter(ax,'A');self.letter(bx,'B')
        ax.set_title('Calibration of held-out probabilities',pad=10)
        bx.set_title('Exploratory decision curve analysis',pad=10)
        for model in MODELS:
            c=cal[cal.model==model].sort_values('bin')
            if c.empty:raise ValueError(f'No calibration data for {model}')
            if c.n_nodes.sum()!=len(data[data.model==model]):raise ValueError(f'Calibration count mismatch {model}')
            lower=c.observed_event_fraction-c.observed_ci_lower
            upper=c.observed_ci_upper-c.observed_event_fraction
            # Percentile intervals can in principle not contain the sample estimate;
            # draw literal CI segments rather than passing invalid negative yerr.
            ax.vlines(c.mean_predicted_probability,c.observed_ci_lower,c.observed_ci_upper,
                      color=COLORS[model],linewidth=.7,alpha=.70)
            ax.plot(c.mean_predicted_probability,c.observed_event_fraction,marker=MARKERS[model],
                    markersize=3.8,color=COLORS[model],ls=STYLES[model],label=LABELS[model])
            z=dca[dca.model==model].sort_values('threshold_probability')
            if z.empty:raise ValueError(f'No DCA data for {model}')
            z=z[z.threshold_probability.between(.05,.50)]
            bx.fill_between(z.threshold_probability,z.ci_lower,z.ci_upper,color=COLORS[model],alpha=.08,lw=0)
            bx.plot(z.threshold_probability,z.net_benefit,color=COLORS[model],ls=STYLES[model],label=LABELS[model])
        ax.plot([0,1],[0,1],ls=':',color='#777777',lw=.85,zorder=0,label='Ideal calibration')
        ax.set(xlim=(0,1),ylim=(-.015,1.015),xlabel='Mean predicted probability',ylabel='Observed malignant-node proportion')
        ax.set_xticks(np.arange(0,1.01,.2));ax.set_yticks(np.arange(0,1.01,.2));ax.set_box_aspect(1)
        for model,label,style in [('TreatAll','Treat all','--'),('TreatNone','Treat none',':')]:
            z=dca[dca.model==model].sort_values('threshold_probability')
            if z.empty:raise ValueError(f'DCA reference missing: {model}')
            z=z[z.threshold_probability.between(.05,.50)]
            bx.plot(z.threshold_probability,z.net_benefit,color='#555555' if model=='TreatAll' else 'black',ls=style,lw=.8,label=label)
        d=dca[dca.threshold_probability.between(.05,.5)]
        plotted=d[d.model.isin(MODELS)]
        lo=min(0,float(d.net_benefit.min()),float(plotted.ci_lower.min()))-.025
        hi=max(.1,float(d.net_benefit.max()),float(plotted.ci_upper.max()))+.025
        bx.set(xlim=(.05,.5),ylim=(lo,hi),xlabel='Threshold probability',ylabel='Net benefit per node')
        bx.set_xticks([.05,.10,.20,.30,.40,.50]);bx.set_box_aspect(1)
        for a in axes:a.legend(loc='upper left',bbox_to_anchor=(-.03,-.20),frameon=False,ncol=2,
                               handlelength=2.0,columnspacing=.8,borderaxespad=0,labelspacing=.5,fontsize=7.1)
        fig.text(.087,.958,'Internal evaluation using source-recorded patient groups',fontsize=8)
        fig.text(.087,.035,'Five quantile bins per model. Intervals use recorded patient group bootstrap of fixed held-out predictions.',fontsize=7.1)
        fig.text(.087,.014,'DCA thresholds 0.05-0.50 are exploratory; they are not prospectively established clinical decision thresholds.',fontsize=7.1)
        self.source(cal,'Figure4_calibration_bins.csv');self.source(dca,'Figure4_decision_curve.csv')
        self.save(fig,'Figure4_Calibration_and_DCA',
                  'Figure 4. Calibration and exploratory node-level decision curve analysis of held-out predictions. A, observed event proportions '
                  'within five probability-quantile bins per model; tied cutpoints may reduce the number of bins. Vertical segments show 95% source-group '
                  'bootstrap confidence intervals conditional on fixed held-out predictions. B, net benefit across exploratory thresholds of 0.05-0.50 '
                  'with recorded patient group bootstrap bands. '
                  'Treat-all and treat-none refer to the selected nodes, and the curves do not establish surgical benefit or validated clinical thresholds.',
                  ['Figure4_calibration_bins.csv','Figure4_decision_curve.csv'])

    @staticmethod
    def feature_label(feature: str) -> str:
        return PRETTY.get(feature,feature.replace('_',' '))

    @staticmethod
    def beeswarm_offsets(values: np.ndarray, width=.30, nbins=35) -> np.ndarray:
        # Deterministic packing within equal-width x bins; all nodes retained.
        v=np.asarray(values,float);bins=np.floor((v-v.min())/(np.ptp(v)+1e-12)*nbins).astype(int)
        offsets=np.zeros(len(v))
        for b in np.unique(bins):
            ii=np.flatnonzero(bins==b);seq=np.arange(len(ii))
            alt=((seq+1)//2)*np.where(seq%2==0,1,-1)
            if len(ii)>1:alt=alt/(np.abs(alt).max()+1e-12)*width
            offsets[ii]=alt
        return offsets

    def figure5(self) -> None:
        shap=self.read_csv('shap_values_long.csv',['node_id','group_id','fold','model','feature','feature_value','shap_value'])
        shap=shap[shap.model=='Combined'].copy()
        if shap.empty or shap.duplicated(['node_id','feature']).any():
            raise ValueError('Expected unique held-out Combined SHAP values for each node-feature')
        d=self.load_predictions();pred=d[d.model=='Combined'].set_index('node_id')
        if set(shap.node_id)!=set(pred.index):raise ValueError('SHAP nodes and held-out prediction nodes do not match')
        identifiers=shap[['node_id','group_id','fold']].drop_duplicates().set_index('node_id')
        if not identifiers.sort_index().equals(pred[['group_id','fold']].sort_index()):
            raise ValueError('SHAP source groups or folds differ from prediction records')
        # Use the real training-fold-imputed input if supplied, retaining the
        # original measurement column unchanged in the source-data export.
        value_column = 'imputed_feature_value' if 'imputed_feature_value' in shap else 'feature_value'
        if shap[[value_column,'shap_value']].isna().any().any():
            raise ValueError('SHAP input must include the actual nonmissing model inputs after training-fold imputation')
        shap=shap.assign(display_feature_value=shap[value_column].to_numpy())
        p=self.analysis/'shap_examples.json'
        if not p.exists():raise FileNotFoundError(p)
        self.inputs[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        spec=json.loads(p.read_text(encoding='utf-8'))
        scale=spec['output_scale'];examples=spec['examples']
        if scale not in ['probability','log_odds']:raise ValueError('output_scale must be probability or log_odds')
        if not examples:raise ValueError('At least one prespecified SHAP example required')
        ex=examples[0];matches=shap[shap.node_id.astype(str)==str(ex['node_id'])].copy()
        if matches.empty:raise ValueError('SHAP example not found in held-out data')
        base=float(ex['base_value']);output=float(ex['model_output']);total=float(matches.shap_value.sum())
        if not np.isclose(base+total,output,atol=1e-5,rtol=1e-5):raise ValueError('Local SHAP additivity failed')
        prob=float(ex['predicted_probability'])
        implied=1/(1+np.exp(-output)) if scale=='log_odds' else output
        if not np.isclose(implied,prob,atol=1e-5,rtol=1e-5):raise ValueError('SHAP output does not reproduce stated probability')
        node=matches.iloc[0].node_id
        if not np.isclose(float(pred.loc[node,'probability']),prob,atol=1e-5):raise ValueError('Example differs from outer-fold prediction')
        if int(pred.loc[node,'fold'])!=int(ex['fold']):raise ValueError('Example fold mismatch')
        rank=shap.groupby('feature',sort=False).shap_value.apply(lambda x:np.mean(np.abs(x))).sort_values(ascending=False)
        top=list(rank.head(12).index);ys=np.arange(len(top))[::-1]
        fig=plt.figure(figsize=(180*MM,166*MM))
        gs=fig.add_gridspec(2,2,left=.27,right=.905,top=.92,bottom=.155,wspace=.22,hspace=.55,height_ratios=[1,1.05])
        ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);cx=fig.add_subplot(gs[1,:])
        self.letter(ax,'A',x=-.71);self.letter(bx,'B',x=-.10);self.letter(cx,'C',x=-.31)
        ax.set_title('Mean absolute contribution',pad=9);bx.set_title('Held-out node contributions',pad=9)
        ax.barh(ys,rank.loc[top].values,color=COLORS['Combined'],height=.68)
        ax.set_yticks(ys,[self.feature_label(x) for x in top]);ax.tick_params(axis='y',length=0)
        ax.set_xlabel('Mean |SHAP value|');ax.set_ylim(-.7,len(top)-.3)
        cmap=LinearSegmentedColormap.from_list('low_high',['#2166AC','#F7F7F7','#B2182B'])
        for feature,y in zip(top,ys):
            g=shap[shap.feature==feature].sort_values('node_id')
            values=g.shap_value.to_numpy(float)
            pct=g.display_feature_value.rank(pct=True,method='average').to_numpy(float)
            scatter=bx.scatter(values,y+self.beeswarm_offsets(values),c=pct,cmap=cmap,vmin=0,vmax=1,s=6,alpha=.77,linewidths=0)
        bx.axvline(0,color='#777777',lw=.6,zorder=0);bx.set_yticks(ys,[]);bx.tick_params(axis='y',length=0)
        bx.set_ylim(ax.get_ylim());bx.set_xlabel('SHAP value');bx.spines.left.set_visible(False)
        cax=fig.add_axes([.927,.603,.009,.28]);bar=fig.colorbar(scatter,cax=cax)
        bar.set_ticks([0,1],labels=['Low','High']);bar.ax.tick_params(labelsize=7,length=0,pad=2)
        bar.set_label('Feature value (within feature)',fontsize=7,labelpad=3);bar.outline.set_visible(False)
        local=matches.reindex(matches.shap_value.abs().sort_values(ascending=False).index)
        parts=[]
        for _,r in local.head(7).iterrows():
            parts.append((self.feature_label(r.feature),float(r.display_feature_value),float(r.shap_value),str(r.feature)))
        rest=local.iloc[7:]
        if len(rest):parts.append((f'Other {len(rest)} features',np.nan,float(rest.shap_value.sum()),'other_features'))
        ypos=np.arange(len(parts))[::-1];current=base;records=[]
        for i,((label,value,contribution,feature),y) in enumerate(zip(parts,ypos)):
            end=current+contribution;color='#B2182B' if contribution>=0 else '#2166AC'
            cx.barh(y,abs(contribution),left=min(current,end),height=.56,color=color,edgecolor='none')
            if i<len(parts)-1:cx.plot([end,end],[y-.29,y-1+.29],color='#999999',lw=.5)
            records.append({'node_id':node,'fold':ex['fold'],'feature':feature,'feature_value':value,'shap_value':contribution,
                            'cumulative_start':current,'cumulative_end':end,'base_value':base,'model_output':output,'predicted_probability':prob,'output_scale':scale})
            current=end
        local_names={'long_axis_mm':'Long axis (mm)','short_axis_mm':'Short axis (mm)',
                     'nc_ct':'Unenhanced CT (HU)','ce_ct':'Enhanced CT (HU)',
                     'ct_70kev':'70-keV CT (HU)','ct_100kev':'100-keV CT (HU)',
                     'nc_ed':'Unenhanced ED (%EDW)','ce_ed':'Enhanced ED (%EDW)',
                     'nc_iodine':'Unenhanced IC (mg/mL)','ce_iodine':'Enhanced IC (mg/mL)'}
        labels=[local_names.get(x[3],x[0])+(f' = {x[1]:.3g}' if np.isfinite(x[1]) else '') for x in parts]
        cx.set_yticks(ypos,labels);cx.tick_params(axis='y',length=0);cx.set_ylim(-.7,len(parts)-.3)
        cx.axvline(base,ls=':',color='#555555',lw=.7);cx.axvline(output,ls='--',color='black',lw=.7)
        scale_label='Log-odds of recorded malignancy' if scale=='log_odds' else 'Predicted probability of recorded malignancy'
        outcome_label='benign' if int(ex['y_true'])==0 else 'malignant'
        cx.set_xlabel(scale_label);cx.set_title(f'Local example: node {node}, fold {ex["fold"]} (recorded {outcome_label})',pad=12,loc='left')
        allpos=np.asarray([base,output]+[r['cumulative_start'] for r in records]+[r['cumulative_end'] for r in records])
        span=max(float(np.ptp(allpos)),.1);cx.set_xlim(allpos.min()-.08*span,allpos.max()+.10*span)
        fig.text(.27,.069,f'Reference output = {base:.3f}    Model output = {output:.3f}    Malignant probability = {prob:.3f}',fontsize=7.4)
        fig.text(.055,.965,f'Combined ridge model; SHAP in {"log-odds" if scale=="log_odds" else "probability"} units',fontsize=8.5)
        fig.text(.055,.022,'Explanations use outer-training backgrounds and held-out nodes. Feature contributions are not causal effects.',fontsize=7.3)
        importance=rank.rename('mean_abs_shap').reset_index()
        importance=importance.assign(included_in_display=importance.feature.isin(top).to_numpy())
        self.source(shap,'Figure5_all_heldout_SHAP_values.csv');self.source(importance,'Figure5_global_importance.csv')
        self.source(pd.DataFrame(records),'Figure5_local_example.csv')
        self.save(fig,'Figure5_Heldout_SHAP',
                  'Figure 5. Explanations of the combined model for held-out nodes. A, the twelve largest mean absolute SHAP contributions. B, the same features ordered '
                  'by mean absolute contribution, with one point per held-out node; colors indicate within-feature ranks of model inputs after '
                  'training-fold-derived imputation. The source CSV retains original measurements and imputed inputs separately. '
                  'C, a recorded benign node selected as closest to probability 0.5 within its outcome class. Contributions add to the model '
                  'output from its outer-training reference value. ED denotes electron density relative to water, and IC denotes iodine concentration. '
                  f'The output scale is {scale}; the displayed probability is checked against the recorded held-out prediction. '
                  'Correlated measurements can share attribution, and SHAP does not establish biological or causal effects.',
                  ['Figure5_all_heldout_SHAP_values.csv','Figure5_global_importance.csv','Figure5_local_example.csv'])

    def supplementary1(self) -> None:
        d=self.read_csv('exploratory_fold_metrics.csv',['algorithm','fold','auc'])
        if d.duplicated(['algorithm','fold']).any():raise ValueError('Duplicate algorithm/fold rows')
        if not d.auc.dropna().between(0,1).all():raise ValueError('Invalid exploratory AUC')
        # Alphabetical, not post hoc "winner" order. All outer folds remain visible.
        algorithms=sorted(d.algorithm.astype(str).unique(),key=str.lower)
        fig,ax=plt.subplots(figsize=(180*MM,104*MM))
        fig.subplots_adjust(left=.235,right=.96,top=.88,bottom=.19)
        yy=np.arange(len(algorithms))[::-1]
        for alg,y in zip(algorithms,yy):
            g=d[d.algorithm.astype(str)==alg].sort_values('fold')
            jitter=np.linspace(-.16,.16,len(g))
            ax.scatter(g.auc,np.full(len(g),y)+jitter,color='#657B89',s=16,alpha=.82,linewidths=.3,edgecolors='white',zorder=3)
            avg=float(g.auc.mean())
            ax.scatter(avg,y,marker='D',s=27,facecolor=COLORS['Combined'],edgecolor='white',linewidth=.5,zorder=4)
        ax.axvline(.5,color='#777777',ls=':',lw=.7)
        ax.set_yticks(yy,algorithms);ax.tick_params(axis='y',length=0)
        xmin = min(.45, float(d.auc.min()) - .03)
        ax.set(xlim=(max(0,xmin),1.005),ylim=(-.6,len(algorithms)-.35),xlabel='Held-out outer-fold ROC-AUC')
        ax.set_xticks(np.arange(.5,1.01,.1))
        ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#657B89',markersize=4,label='Individual outer fold'),
                           Line2D([0],[0],marker='D',color='none',markerfacecolor=COLORS['Combined'],markersize=4.5,label='Mean across outer folds')],
                  frameon=False,loc='lower right',bbox_to_anchor=(1,-.23),ncol=2)
        fig.text(.06,.955,'Exploratory algorithm comparison',fontsize=9)
        fig.text(.06,.035,'Each point is one outer fold. Fold estimates are dependent; no fold-based significance tests are performed.',fontsize=7.4)
        self.source(d,'SupplementaryFigure1_fold_metrics.csv')
        means=d.groupby('algorithm',sort=True).agg(mean_auc=('auc','mean'),n_folds=('fold','nunique')).reset_index()
        self.source(means,'SupplementaryFigure1_algorithm_means.csv')
        self.save(fig,'SupplementaryFigure1_Exploratory_algorithm_AUC',
                  'Supplementary Figure 1. ROC-AUC for the exploratory algorithms in each outer evaluation fold. Diamonds show the mean across folds '
                  'and circles show individual folds. Algorithm order is alphabetical. The folds are not treated as independent replicates and no '
                  'significance tests are based on the five fold estimates. This comparison does not replace the prespecified three-model primary analysis.',
                  ['SupplementaryFigure1_fold_metrics.csv','SupplementaryFigure1_algorithm_means.csv'])

    def finish(self) -> None:
        oldpath=self.figdir/'figure_manifest.json'
        old=json.loads(oldpath.read_text(encoding='utf-8')) if oldpath.exists() else {'figures':[]}
        previous={x['figure']:x for x in old.get('figures',[])}
        previous.update({x['figure']:x for x in self.manifest})
        payload={'python':platform.python_version(),'matplotlib':matplotlib.__version__,'backend':matplotlib.get_backend(),
                 'font':'Arial','raster_dpi':300,'width_mm':180,'input_sha256':{**old.get('input_sha256',{}),**self.inputs},
                 'figures':sorted(previous.values(),key=lambda x:x['figure'])}
        oldpath.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
        (self.figdir/'figure_captions.txt').write_text('\n\n'.join(x['caption'] for x in payload['figures'])+'\n',encoding='utf-8')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--figures',nargs='+',choices=['1','3','4','5','S1'],default=['1','3','4','5','S1'])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    setup_style();p=Plotter(args.root)
    actions={'1':p.figure1,'3':p.figure3,'4':p.figure4,'5':p.figure5,'S1':p.supplementary1}
    for f in args.figures:actions[f]()
    p.finish()


if __name__=='__main__':
    main()
