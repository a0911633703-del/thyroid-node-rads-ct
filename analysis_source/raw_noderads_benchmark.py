"""Secondary direct-score Node-RADS benchmark on the fixed primary cohort.

No model fitting, threshold selection, or primary-model modification is performed.
Source-group bootstrap intervals condition on the supplied combined OOF predictions.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def percentile_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'analysis')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260926)
    args = parser.parse_args()
    directory = args.analysis_dir.resolve()
    cohort = pd.read_csv(directory / 'cohort_primary.csv', dtype={'node_id': str, 'group_id': str})
    predictions = pd.read_csv(directory / 'primary_predictions.csv', dtype={'node_id': str, 'group_id': str})
    assert len(cohort) == 248 and cohort.node_id.nunique() == 248
    assert cohort.group_id.nunique() == 130 and cohort.outcome.sum() == 84
    assert cohort.node_rads.isin([1, 2, 3, 4, 5]).all()
    models = {}
    for model in ['NodeRADS', 'Combined']:
        frame = predictions.loc[predictions.model == model].set_index('node_id').loc[cohort.node_id]
        assert frame.index.is_unique
        assert np.array_equal(frame.group_id.to_numpy(), cohort.group_id.to_numpy())
        assert np.array_equal(frame.y_true.to_numpy(), cohort.outcome.to_numpy())
        assert np.array_equal(frame.fold.to_numpy(), cohort.outer_fold.to_numpy())
        models[model] = frame.probability.to_numpy()
    y = cohort.outcome.to_numpy(dtype=int)
    score = cohort.node_rads.to_numpy(dtype=float)
    groups = np.array(sorted(cohort.group_id.unique()))
    group_index = {group: i for i, group in enumerate(groups)}
    group_codes = np.array([group_index[group] for group in cohort.group_id])
    rng = np.random.default_rng(args.seed)
    bootstrap = np.full((args.bootstrap, 3), np.nan)
    estimate = np.array([roc_auc_score(y, score), average_precision_score(y, score),
                         roc_auc_score(y, models['Combined']) - roc_auc_score(y, score)])
    for b in range(args.bootstrap):
        counts = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        weight = counts[group_codes]
        positives = float(np.dot(weight, y))
        negatives = float(weight.sum() - positives)
        if positives > 0 and negatives > 0:
            raw_auc = roc_auc_score(y, score, sample_weight=weight)
            bootstrap[b] = [raw_auc, average_precision_score(y, score, sample_weight=weight),
                            roc_auc_score(y, models['Combined'], sample_weight=weight) - raw_auc]
    fold_raw, fold_lr = [], []
    for fold in sorted(cohort.outer_fold.unique()):
        take = cohort.outer_fold.to_numpy() == fold
        fold_raw.append(roc_auc_score(y[take], score[take]))
        fold_lr.append(roc_auc_score(y[take], models['NodeRADS'][take]))
    assert np.allclose(fold_raw, fold_lr, atol=1e-12), 'Fold-wise LR does not preserve raw-score ranking'
    score_probability_variation = {
        int(category): int(len(np.unique(models['NodeRADS'][score == category])))
        for category in sorted(set(score))
    }
    rows = []
    names = [('RawNodeRADS', 'roc_auc'), ('RawNodeRADS', 'average_precision'),
             ('Combined_minus_RawNodeRADS', 'roc_auc_difference')]
    for i, (benchmark, metric) in enumerate(names):
        low, high, valid = percentile_interval(bootstrap[:, i])
        rows.append({'benchmark': benchmark, 'metric': metric, 'estimate': float(estimate[i]),
                     'ci_lower': low, 'ci_upper': high, 'bootstrap_valid': valid,
                     'n_nodes': len(y), 'n_source_groups': len(groups), 'malignant_nodes': int(y.sum()),
                     'analysis_role': 'secondary_descriptive_no_P_value',
                     'interval_scope': 'source_group_bootstrap_fixed_combined_OOF_predictions',
                     'bootstrap_replicates': args.bootstrap, 'bootstrap_seed': args.seed,
                     'score': 'original_ordinal_NodeRADS_1_to_5_higher_means_more_suspicious',
                     'foldwise_raw_LR_AUC_equal': True,
                     'raw_mean_fold_AUC': float(np.mean(fold_raw)),
                     'LR_pooled_OOF_AUC': float(roc_auc_score(y, models['NodeRADS']))})
    result = pd.DataFrame(rows)
    result.to_csv(directory / 'raw_noderads_benchmark.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    print(result[['benchmark', 'metric', 'estimate', 'ci_lower', 'ci_upper', 'bootstrap_valid']].to_string(index=False))
    print(json.dumps({'fold_raw_auc': fold_raw, 'fold_LR_auc': fold_lr,
                      'different_LR_probabilities_per_raw_grade': score_probability_variation,
                      'interpretation': 'Within each fold, positive-slope logistic regression preserves raw-score ranking and ties. Pooling probabilities from different fitted fold models changes cross-fold ties and comparisons. The unweighted mean of within-fold AUCs also weights case-control pairs differently from pooled raw AUC; these two numerical differences must not be attributed entirely to imaging features.'}))


if __name__ == '__main__':
    main()
