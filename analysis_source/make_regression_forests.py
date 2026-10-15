"""Exploratory OR forests for the recorded binary node outcome.

Run with the recorded Python environment. This script preserves the frozen primary
analysis and fits neither Cox models nor replacement prediction models.
One source group is the resampling cluster; its hospital identity is unverified.
"""
from __future__ import annotations

import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from scipy.stats import norm
import scipy
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
import statsmodels
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261008
FEATURES = ['nc_ct', 'ce_ct', 'nc_iodine', 'ce_iodine', 'nc_zeff', 'ce_zeff',
            'nc_ed', 'ce_ed', 'ct_70kev', 'ct_100kev', 'casupp_30', 'casupp_70',
            'short_axis_mm', 'long_axis_mm', 'node_rads']
ORDER = ['node_rads', 'nc_ct', 'ce_ct', 'nc_iodine', 'ce_iodine', 'nc_zeff',
         'ce_zeff', 'nc_ed', 'ce_ed', 'ct_70kev', 'ct_100kev', 'casupp_30',
         'casupp_70', 'short_axis_mm', 'long_axis_mm']
EXTRA = ['delta_ct', 'delta_iodine', 'spectral_slope_70_100', 'delta_casupp',
         'ce_ct_artery', 'ce_ct_muscle', 'ct_ratio_artery', 'ct_ratio_muscle']
UNIVARIABLE_ORDER = ORDER + EXTRA
NAMES = {
    'node_rads': 'Node-RADS', 'nc_ct': 'Unenhanced attenuation',
    'ce_ct': 'Enhanced attenuation', 'nc_iodine': 'Unenhanced iodine',
    'ce_iodine': 'Enhanced iodine', 'nc_zeff': 'Unenhanced Zeff',
    'ce_zeff': 'Enhanced Zeff', 'nc_ed': 'Unenhanced electron density',
    'ce_ed': 'Enhanced electron density', 'ct_70kev': '70-keV attenuation',
    'ct_100kev': '100-keV attenuation', 'casupp_30': 'Ca-suppression setting 30',
    'casupp_70': 'Ca-suppression setting 70',
    'short_axis_mm': 'Short-axis diameter', 'long_axis_mm': 'Long-axis diameter',
    'delta_ct': 'Attenuation increment', 'delta_iodine': 'Iodine increment',
    'spectral_slope_70_100': '70-to-100-keV slope', 'delta_casupp': 'Ca-suppression difference',
    'ce_ct_artery': 'Arterial reference attenuation', 'ce_ct_muscle': 'Muscle reference attenuation',
    'ct_ratio_artery': 'Node / artery CT ratio', 'ct_ratio_muscle': 'Node / muscle CT ratio',
}
UNITS = {x: 'HU' for x in ['nc_ct', 'ce_ct', 'ct_70kev', 'ct_100kev', 'casupp_30', 'casupp_70']}
UNITS.update({'nc_iodine': 'mg/mL', 'ce_iodine': 'mg/mL', 'nc_zeff': 'dimensionless',
              'ce_zeff': 'dimensionless', 'nc_ed': '%EDW', 'ce_ed': '%EDW',
              'short_axis_mm': 'mm', 'long_axis_mm': 'mm', 'node_rads': 'grade'})
UNITS.update({'delta_ct': 'HU', 'delta_iodine': 'mg/mL', 'spectral_slope_70_100': 'HU/keV',
              'delta_casupp': 'HU', 'ce_ct_artery': 'HU', 'ce_ct_muscle': 'HU',
              'ct_ratio_artery': 'dimensionless', 'ct_ratio_muscle': 'dimensionless'})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def initialize(root: Path, replicates: int):
    an = root / 'analysis' / 'extended_regression_20261008'
    an.mkdir(parents=True, exist_ok=True)
    cohort_path = root / 'analysis' / 'cohort_primary.csv'
    model_path = root / 'analysis' / 'models' / 'Combined_full_development_model.joblib'
    cohort = pd.read_csv(cohort_path)
    assert len(cohort) == 248 and cohort.group_id.nunique() == 130
    assert cohort.node_id.is_unique and int(cohort.outcome.sum()) == 84
    assert set(cohort.outcome.unique()) == {0, 1}
    assert cohort[FEATURES].isna().sum().sum() == 1 and cohort.nc_zeff.isna().sum() == 1
    assert cohort[EXTRA].notna().all().all()
    assert np.allclose(cohort.delta_ct, cohort.ce_ct - cohort.nc_ct)
    assert np.allclose(cohort.delta_iodine, cohort.ce_iodine - cohort.nc_iodine)
    assert np.allclose(cohort.spectral_slope_70_100, (cohort.ct_70kev - cohort.ct_100kev) / 30)
    assert np.allclose(cohort.delta_casupp, cohort.casupp_70 - cohort.casupp_30)
    assert np.allclose(cohort.ct_ratio_artery, cohort.ce_ct / cohort.ce_ct_artery)
    assert np.allclose(cohort.ct_ratio_muscle, cohort.ce_ct / cohort.ce_ct_muscle)
    scales = {x: 1.0 if x == 'node_rads' else float(cohort[x].std(ddof=0)) for x in UNIVARIABLE_ORDER}
    assert all(np.isfinite(v) and v > 0 for v in scales.values())
    spec = {
        'analysis_role': 'post-review exploratory association/model-coefficient displays; no feature screening',
        'outcome': 'recorded binary node status from original homogeneous cervical-level pathology assignment',
        'cohort': {'nodes': 248, 'source_groups': 130, 'malignant_nodes': 84},
        'grouping': 'source-recorded patient group; hospital identities remain unverified',
        'adjusted_model_features': FEATURES, 'univariable_variables': UNIVARIABLE_ORDER,
        'univariable_primary_inputs': ORDER, 'univariable_secondary_variables': EXTRA,
        'random_seed': SEED,
        'effect_units': 'one fixed full-primary-cohort observed-value population SD (ddof=0) per continuous variable; Node-RADS per grade',
        'fixed_reference_increments': scales,
        'univariable': 'one-predictor binomial-logit GEE; Independence working correlation; robust source-group sandwich SE; two-sided normal-Wald CI/P; missing value omitted only for that predictor; BH across all 23 P values, including 8 secondary recorded/derived variables',
        'invalid_legacy_ratios': 'mixed-unit iodine/HU and electron-density/HU ratios excluded; not called normalized iodine or electron density',
        'adjusted': 'saved full-development combined ridge logistic model C=1; all 15 inputs retained; no new C selection',
        'bootstrap': {'replicates': replicates, 'sampling': '130 source groups sampled with replacement, concatenating all node records per sampled group including duplicate draws',
                      'refit': 'clone saved pipeline; fixed C=1; refit median imputation and StandardScaler in every sample',
                      'interval': '2.5th/97.5th percentiles of original-unit coefficients times fixed reference increments, then exponentiation',
                      'scope': 'descriptive conditional on fixed C and reference increments; no hypothesis P values; not validation performance or causal effects'},
        'threads': {'BLAS': 1, 'workers': 1},
        'source_hashes': {'cohort_primary.csv': sha(cohort_path), 'Combined_full_development_model.joblib': sha(model_path),
                          'analysis_plan_frozen.json': sha(root / 'analysis_plan_frozen.json')},
        'method_sources': ['https://www.statsmodels.org/stable/gee.html',
                           'https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_estimating_equations.GEE.fit.html',
                           'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'],
        'source_checked_date': '2026-10-08',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__, 'sklearn': sklearn.__version__,
                     'matplotlib': matplotlib.__version__, 'joblib': joblib.__version__},
    }
    plan_path = an / 'extended_plan_before_results.json'
    if plan_path.exists():
        prior = json.loads(plan_path.read_text(encoding='utf-8'))
        if prior['specification'] != spec:
            raise ValueError('An extended plan already exists with different parameters; preserve it rather than overwrite.')
    else:
        json_write(plan_path, {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'specification': spec})
    print('Extended exploratory plan recorded before computation.', flush=True)
    return an, cohort, joblib.load(model_path), scales, spec


def gee_univariable(cohort: pd.DataFrame, scales: dict, an: Path) -> pd.DataFrame:
    rows = []
    invalid = []
    for feature in UNIVARIABLE_ORDER:
        frame = cohort.dropna(subset=[feature])
        delta = scales[feature]
        mean = float(cohort[feature].mean())
        x = (frame[feature].to_numpy(float) - mean) / delta
        exog = sm.add_constant(x, has_constant='add')
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = sm.GEE(frame.outcome.to_numpy(float), exog,
                            groups=frame.group_id.to_numpy(),
                            family=sm.families.Binomial(link=sm.families.links.Logit()),
                            cov_struct=sm.cov_struct.Independence()).fit(maxiter=200, ctol=1e-10, cov_type='robust')
        beta = float(result.params[1])
        se = float(result.bse[1])
        converged = bool(result.converged)
        status = 'valid' if converged and np.isfinite(beta) and np.isfinite(se) and se > 0 else 'invalid'
        if status != 'valid':
            invalid.append(feature)
        limits = beta + np.array([-1, 1]) * norm.ppf(.975) * se
        rows.append({'feature': feature, 'variable': NAMES[feature],
                     'predictor_role': 'Primary model input' if feature in FEATURES else 'Secondary descriptive variable',
                     'n_nodes': len(frame),
                     'n_source_groups': int(frame.group_id.nunique()), 'malignant_nodes': int(frame.outcome.sum()),
                     'reference_increment': delta, 'original_unit': UNITS[feature],
                     'comparison': 'per grade' if feature == 'node_rads' else 'per fixed full-cohort SD',
                     'beta_log_odds_per_increment': beta, 'robust_cluster_se': se,
                     'odds_ratio': float(np.exp(beta)), 'ci_lower': float(np.exp(limits[0])),
                     'ci_upper': float(np.exp(limits[1])), 'p_two_sided_wald': float(result.pvalues[1]),
                     'converged': converged, 'fit_status': status, 'fit_seconds': time.perf_counter() - start,
                     'warnings': ' | '.join(sorted(set(str(w.message) for w in caught)))})
    out = pd.DataFrame(rows)
    csv_write(an / 'univariable_GEE_OR.csv', out)
    if invalid:
        json_write(an / 'univariable_invalid.json', {'features': invalid})
        raise RuntimeError('Invalid GEE coefficients/SE/convergence; recorded and halted: ' + ', '.join(invalid))
    from statsmodels.stats.multitest import multipletests
    adjusted_p = multipletests(out['p_two_sided_wald'].to_numpy(copy=True), alpha=.05, method='fdr_bh')[1]
    out.loc[:, 'p_BH_23_variables'] = adjusted_p
    assert np.isfinite(out[['odds_ratio', 'ci_lower', 'ci_upper', 'p_two_sided_wald', 'p_BH_23_variables']].to_numpy()).all()
    csv_write(an / 'univariable_GEE_OR.csv', out)
    print('All 23 univariable GEE fits converged with positive robust SE.', flush=True)
    return out


def original_coefficients(pipe):
    scaler = pipe.named_steps['scaler']
    model = pipe.named_steps['model']
    beta = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - np.dot(beta, scaler.mean_))
    return beta, intercept


def ridge_adjusted(cohort: pd.DataFrame, saved, scales: dict, an: Path, replicates: int):
    assert list(saved.feature_names_in_) == FEATURES
    assert saved.named_steps['model'].C == 1
    X, y = cohort[FEATURES], cohort.outcome
    reference_delta = np.array([scales[x] for x in FEATURES])
    beta, intercept = original_coefficients(saved)
    imputed = saved.named_steps['imputer'].transform(X)
    saved_probability = saved.predict_proba(X)[:, 1]
    formula_probability = expit(intercept + imputed @ beta)
    formula_error = float(np.max(np.abs(saved_probability - formula_probability)))
    assert formula_error < 1e-10
    reproduced = clone(saved).fit(X, y)
    refit_error = float(np.max(np.abs(saved_probability - reproduced.predict_proba(X)[:, 1])))
    assert refit_error < 1e-8, 'Same-C full-cohort refit failed to reproduce saved model.'
    csv_write(an / 'saved_full_model_probability_reproduction.csv', pd.DataFrame({
        'node_id': cohort.node_id, 'recorded_outcome': y, 'saved_probability': saved_probability,
        'formula_probability': formula_probability, 'same_C_refit_probability': reproduced.predict_proba(X)[:, 1]}))
    groups = cohort.group_id.unique()
    indices = [np.flatnonzero(cohort.group_id.to_numpy() == g) for g in groups]
    rng = np.random.default_rng(SEED)
    coefficients, logs, draws = [], [], []
    start = time.perf_counter()
    for b in range(replicates):
        sampled = rng.integers(0, len(groups), len(groups))
        ix = np.concatenate([indices[j] for j in sampled])
        sample = cohort.iloc[ix]
        status, error, warning_text = 'valid', '', ''
        row = np.full(len(FEATURES), np.nan)
        began = time.perf_counter()
        try:
            if sample.outcome.nunique() != 2:
                raise ValueError('One outcome class in resample')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = clone(saved).fit(sample[FEATURES], sample.outcome)
            warning_text = ' | '.join(sorted(set(str(w.message) for w in caught)))
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise ValueError('ConvergenceWarning in fixed-C bootstrap fit')
            beta_b, _ = original_coefficients(model)
            row = beta_b * reference_delta
            if not np.isfinite(row).all():
                raise ValueError('Nonfinite mapped coefficient')
        except Exception as exc:
            status, error = 'invalid', str(exc)
            row[:] = np.nan
        coefficients.append(row)
        draws.append({'replicate': b + 1, **{str(groups[j]): int(np.count_nonzero(sampled == j)) for j in range(len(groups))}})
        logs.append({'replicate': b + 1, 'status': status, 'error': error, 'warnings': warning_text,
                     'n_nodes': len(sample), 'n_distinct_drawn_source_groups': int(np.unique(sampled).size),
                     'malignant_nodes': int(sample.outcome.sum()), 'seconds': time.perf_counter() - began})
        if (b + 1) % 250 == 0:
            print(f'Fixed-C source-group bootstrap: {b + 1}/{replicates}', flush=True)
    elapsed = time.perf_counter() - start
    boot = pd.DataFrame(coefficients, columns=FEATURES)
    boot.insert(0, 'replicate', np.arange(1, replicates + 1))
    log = pd.DataFrame(logs)
    csv_write(an / 'ridge_bootstrap_log_odds_per_fixed_increment.csv', boot)
    csv_write(an / 'ridge_bootstrap_fit_log.csv', log)
    csv_write(an / 'ridge_bootstrap_group_draw_counts.csv', pd.DataFrame(draws))
    valid = log.status.eq('valid').to_numpy()
    n_valid = int(valid.sum())
    if n_valid < int(np.ceil(.95 * replicates)):
        raise RuntimeError(f'Only {n_valid}/{replicates} bootstrap fits valid; do not present stable intervals.')
    qs = np.percentile(np.array(coefficients)[valid], [2.5, 97.5], axis=0)
    mapped = beta * reference_delta
    rows = []
    for feature in ORDER:
        j = FEATURES.index(feature)
        rows.append({'feature': feature, 'variable': NAMES[feature], 'n_nodes': len(cohort),
                     'n_source_groups': len(groups), 'malignant_nodes': int(y.sum()),
                     'reference_increment': scales[feature], 'original_unit': UNITS[feature],
                     'comparison': 'per grade' if feature == 'node_rads' else 'per fixed full-cohort SD',
                     'coefficient_original_units': float(beta[j]), 'beta_log_odds_per_increment': float(mapped[j]),
                     'odds_ratio': float(np.exp(mapped[j])), 'ci_lower': float(np.exp(qs[0, j])),
                     'ci_upper': float(np.exp(qs[1, j])), 'bootstrap_valid': n_valid, 'bootstrap_total': replicates,
                     'bootstrap_fraction_beta_positive': float((np.array(coefficients)[valid, j] > 0).mean()),
                     'fixed_C': 1.0, 'interval_role': 'descriptive fixed-C source-group percentile bootstrap'})
    out = pd.DataFrame(rows)
    assert np.isfinite(out[['odds_ratio', 'ci_lower', 'ci_upper']].to_numpy()).all()
    csv_write(an / 'adjusted_fixed_C_ridge_OR.csv', out)
    report = {'bootstrap_replicates': replicates, 'bootstrap_valid': n_valid,
              'bootstrap_invalid': replicates - n_valid, 'bootstrap_seconds': elapsed,
              'saved_formula_max_abs_probability_error': formula_error,
              'same_C_refit_max_abs_probability_error': refit_error,
              'imputed_primary_measurements': int(X.isna().sum().sum()),
              'reference_increments_fixed': True, 'new_hyperparameter_search': False,
              'ridge_P_values': 'not calculated'}
    json_write(an / 'ridge_bootstrap_report.json', report)
    print(json.dumps(report), flush=True)
    return out, report


def fmt(x: float) -> str:
    return f'{x:.2f}' if .01 <= x < 100 else f'{x:.2g}'


def p_fmt(x: float) -> str:
    return f'{x:.3f}' if x >= .001 else f'{x:.1e}'


def unit_text(row) -> str:
    if row.feature == 'node_rads':
        return '1 grade'
    unit = '' if row.original_unit == 'dimensionless' else ' ' + row.original_unit
    return f'1 SD ({row.reference_increment:.3g}{unit})'


def plots(root: Path, an: Path, univ: pd.DataFrame, adjusted: pd.DataFrame, report: dict):
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.5, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'aizhaodong_OR_20261008',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'axes.unicode_minus': False})
    figdir = root / 'figures_tables' / 'figures'
    srcdir = root / 'figures_tables' / 'source_data'
    tabdir = root / 'figures_tables' / 'tables'
    for d in [figdir, srcdir, tabdir]:
        d.mkdir(parents=True, exist_ok=True)
    manifests = []
    for data, stem, is_univ in [(univ, 'SupplementaryFigure6_Univariable_OR_forest', True),
                                (adjusted, 'SupplementaryFigure7_Adjusted_ridge_OR_forest', False)]:
        fig = plt.figure(figsize=(180 / 25.4, (220 if is_univ else 146) / 25.4))
        y_max = 28 if is_univ else 16
        if is_univ:
            left, width, numeric, p_pos, q_pos = .433, .221, .671, .874, .955
        else:
            left, width, numeric, p_pos, q_pos = .433, .315, .782, None, None
        table = fig.add_axes([.0, .15, 1., .76])
        table.set_xlim(0, 1)
        table.set_ylim(-.7, y_max)
        table.axis('off')
        forest = fig.add_axes([left, .15, width, .76], facecolor='none')
        forest.set_ylim(-.7, y_max)
        lo = float(data.ci_lower.min())
        hi = float(data.ci_upper.max())
        xlim = [10 ** (np.floor(np.log10(lo) * 2) / 2), 10 ** (np.ceil(np.log10(hi) * 2) / 2)]
        forest.set_xscale('log')
        forest.set_xlim(xlim)
        forest.axvline(1, color='#777777', linestyle='--', lw=.8, zorder=1)
        for spine in ['top', 'left', 'right']:
            forest.spines[spine].set_visible(False)
        forest.spines['bottom'].set_linewidth(.6)
        forest.set_yticks([])
        forest.tick_params(axis='x', labelsize=7.5, length=2.5, width=.6)
        from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator
        forest.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        forest.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        forest.xaxis.set_minor_locator(NullLocator())
        forest.set_xlabel('Odds ratio (log scale)', fontsize=7.5, labelpad=3)
        header_y = 27.1 if is_univ else 15.4
        table.text(.018, header_y, 'Variable', weight='bold', va='center', fontsize=8)
        table.text(.395, header_y, 'n', weight='bold', ha='center', va='center', fontsize=8)
        table.text(numeric, header_y, 'OR (95% CI)', weight='bold', va='center', fontsize=8)
        if is_univ:
            table.text(p_pos, header_y, 'P', weight='bold', ha='center', va='center', fontsize=8)
            table.text(q_pos, header_y, 'BH q', weight='bold', ha='center', va='center', fontsize=8)
            table.text(.018, 25.5, 'Primary model inputs', weight='bold', va='center', fontsize=8)
            table.text(.018, 8.5, 'Secondary recorded or derived variables', weight='bold', va='center', fontsize=8)
        for i, row in enumerate(data.itertuples()):
            y = (24 - i if i < 15 else 22 - i) if is_univ else 14 - i
            if i % 2 == 0:
                table.add_patch(Rectangle((.01, y - .48), .98, .96, facecolor='#F3F5F5', edgecolor='none', zorder=0))
            label_unit = '' if row.original_unit in ['dimensionless', 'grade'] else f' ({row.original_unit})'
            table.text(.018, y, row.variable + label_unit, va='center', fontsize=7.5)
            table.text(.395, y, str(row.n_nodes), va='center', ha='center', fontsize=7.5)
            table.text(numeric, y, f'{fmt(row.odds_ratio)} ({fmt(row.ci_lower)}, {fmt(row.ci_upper)})', va='center', fontsize=7.5)
            if is_univ:
                table.text(p_pos, y, p_fmt(row.p_two_sided_wald), ha='center', va='center', fontsize=7.5)
                table.text(q_pos, y, p_fmt(row.p_BH_23_variables), ha='center', va='center', fontsize=7.5)
            forest.plot([row.ci_lower, row.ci_upper], [y, y], color='#333333', lw=.8, zorder=3)
            forest.scatter([row.odds_ratio], [y], marker='s', s=14, color='#333333' if is_univ else '#009E73', zorder=4)
        title = 'Univariable associations with recorded node malignancy' if is_univ else 'Adjusted coefficients of the full-development combined ridge model'
        fig.text(.018, .964, title, fontsize=8, ha='left', va='top')
        footer = ['Continuous measurements: one fixed full-cohort SD. Node-RADS: one grade.',
                  '248 selected nodes in 130 source-recorded groups; hospital patient identities unverified.',
                  'Binomial-logit GEE with robust group SE; BH across all 23 comparisons. No variable screening.' if is_univ
                  else f'Fixed C = 1; {report["bootstrap_valid"]:,}/{report["bootstrap_replicates"]:,} group-bootstrap refits; descriptive percentile 95% intervals.']
        for y, text in zip([.078, .053, .028], footer):
            fig.text(.018, y, text, fontsize=7.5, ha='left', va='center')
        files = []
        for ext in ['pdf', 'svg', 'png', 'tif']:
            path = figdir / f'{stem}.{ext}'
            kwargs = {'dpi': 300, 'facecolor': 'white'}
            if ext == 'tif':
                kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
            fig.savefig(path, **kwargs)
            files.append({'path': str(path.relative_to(root)), 'sha256': sha(path)})
        plt.close(fig)
        csv_write(srcdir / f'{stem}_source.csv', data)
        with Image.open(figdir / f'{stem}.png') as im:
            raster = {'pixels': list(im.size), 'dpi': list(im.info.get('dpi', []))}
        manifests.append({'stem': stem, 'files': files, 'raster': raster, 'rows': len(data),
                          'forest_limits': xlim, 'CI_clipped': False, 'editable_svg_text': '<text' in (figdir / f'{stem}.svg').read_text()})
    tables = []
    for data, num, is_univ in [(univ, 8, True), (adjusted, 9, False)]:
        headers = ['Variable', 'Comparison increment', 'n', 'OR (95% CI)'] + (['P', 'BH q', 'Role'] if is_univ else [])
        rows = []
        for row in data.itertuples():
            cells = [row.variable, unit_text(row), str(row.n_nodes),
                     f'{fmt(row.odds_ratio)} ({fmt(row.ci_lower)} to {fmt(row.ci_upper)})']
            if is_univ:
                cells += [p_fmt(row.p_two_sided_wald), p_fmt(row.p_BH_23_variables), 'Primary' if row.feature in FEATURES else 'Secondary']
            rows.append(cells)
        notes = ['Continuous-variable comparisons use a fixed observed-value full-primary-cohort population SD (ddof = 0), excluding the one original missing value only when computing its reference SD. Node-RADS is per one grade.',
                 'The outcome is recorded node status under cervical-level pathological assignment. Source patient groups remain unverified against hospital identities. These exploratory displays did not select predictors or replace the prespecified models.']
        if is_univ:
            notes += ['Each one-predictor binomial-logit GEE used Independence working correlation and robust source-group sandwich covariance. Confidence intervals and two-sided P values use the normal-Wald approximation. BH q values adjust all 23 univariable comparisons. Eight secondary recorded/derived variables are displayed separately from the 15 primary model inputs. They do not enter the primary adjusted model. Mixed-unit iodine/HU and electron-density/HU legacy ratios are not used.',
                      'The one missing unenhanced Zeff value is omitted only from its univariable analysis (n = 247). All other analyses have n = 248.']
        else:
            notes += [f'Adjusted ORs are exponentiated saved full-development combined ridge coefficients mapped into these fixed increments. All 15 inputs remain in the model, with fixed C = 1. The one original missing measurement was median-imputed.',
                      f'Descriptive 95% percentile intervals use {report["bootstrap_valid"]} valid fits from {report["bootstrap_replicates"]} source-group resamples, refitting the median imputer, scaling and fixed-C ridge model in every sample. Hyperparameters were not retuned. These intervals condition on C and are not unpenalized Wald intervals or independently validated causal effects. No ridge P values were calculated.']
        tables.append({'id': f'SupplementaryTable{num}', 'title': 'Exploratory univariable GEE associations with recorded node malignancy' if is_univ else 'Exploratory adjusted full-development ridge odds ratios',
                       'headers': headers, 'rows': rows, 'notes': notes, 'supplementary': True,
                       'orientation': 'portrait', 'font_size': 9,
                       'column_widths_mm': [40, 28, 10, 37, 18, 18, 17] if is_univ else [60, 38, 14, 56],
                       'column_alignments': ['left', 'left', 'center', 'center'] + (['center', 'center', 'center'] if is_univ else [])})
        csv_write(tabdir / f'SupplementaryTable{num}_regression_source.csv', data)
        csv_write(tabdir / f'SupplementaryTable{num}_formatted.csv', pd.DataFrame(rows, columns=headers))
    json_write(tabdir / 'extended_regression_tables.json', tables)
    captions = {
        'SupplementaryFigure6_Univariable_OR_forest': 'Exploratory univariable associations with recorded node malignancy. Points are odds ratios and horizontal lines are normal-Wald 95% confidence intervals from separate binomial-logit GEE models with Independence working correlation and robust source-group covariance. Fifteen primary model inputs and eight secondary recorded/derived variables are shown in separate row blocks. Continuous measurements are per one fixed primary-cohort observed-value population SD, and Node-RADS is per grade. The primary table has 248 selected nodes in 130 source-recorded groups. The original missing unenhanced Zeff measurement yields n = 247 only for that predictor. Two-sided P values and BH q values for all 23 comparisons are displayed. Predictor screening was not performed, and hospital identities remain unverified. Mixed-unit legacy iodine/HU and electron-density/HU ratios are excluded.',
        'SupplementaryFigure7_Adjusted_ridge_OR_forest': f'Exploratory adjusted coefficients of the full-development combined ridge logistic model. Points are exponentiated saved coefficients mapped to one fixed full-primary-cohort population SD for continuous measurements or one Node-RADS grade. Horizontal lines are descriptive percentile 95% intervals from {report["bootstrap_valid"]} valid fixed-C fits among {report["bootstrap_replicates"]} source-group bootstrap resamples. Each resample concatenated all nodes from each of 130 sampled source groups, including repeated group draws, and refitted training median imputation, scaling and ridge coefficients with C = 1. The intervals condition on the selected C and do not include a repeated tuning procedure. No ridge P values were calculated. Estimates describe the model fitted to all 248 selected nodes and are neither held-out performance estimates nor causal effects. Recorded hospital identity linkage remains unverified.'}
    json_write(an / 'forest_captions.json', captions)
    json_write(an / 'forest_manifest.json', manifests)
    return manifests, tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--stage', choices=['all', 'analysis', 'plot'], default='all')
    args = parser.parse_args()
    if args.bootstrap_replicates < 2000:
        raise ValueError('Final descriptive bootstrap requires at least 2000 source-group resamples.')
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        an, cohort, saved, scales, spec = initialize(args.root, args.bootstrap_replicates)
        if args.stage != 'plot':
            univ = gee_univariable(cohort, scales, an)
            adjusted, report = ridge_adjusted(cohort, saved, scales, an, args.bootstrap_replicates)
        else:
            univ = pd.read_csv(an / 'univariable_GEE_OR.csv')
            adjusted = pd.read_csv(an / 'adjusted_fixed_C_ridge_OR.csv')
            report = json.loads((an / 'ridge_bootstrap_report.json').read_text())
        if args.stage != 'analysis':
            plots(args.root, an, univ, adjusted, report)
    for name, expected in spec['source_hashes'].items():
        path = args.root / ('analysis' if name == 'cohort_primary.csv' else 'analysis/models' if name.endswith('.joblib') else '') / name
        assert sha(path) == expected, 'Original primary input or model was modified: ' + name
    json_write(an / 'extended_run_report.json', {'stage': args.stage, 'total_seconds': time.perf_counter() - start,
                                               'primary_input_hashes_unchanged': True,
                                               'fit_report': report})
    print(f'Finished extended {args.stage} stage in {time.perf_counter()-start:.2f} seconds.', flush=True)


if __name__ == '__main__':
    main()
