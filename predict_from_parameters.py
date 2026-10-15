"""Calculate saved full-development node probabilities with standard Python.

This returns probabilities only. It does not impose a clinical threshold and
does not recreate the separate outer-fold OOF evaluation predictions.
"""
from pathlib import Path
import argparse,csv,json,math

def expit_stable(z):
    if z>=0:return 1.0/(1.0+math.exp(-z))
    e=math.exp(z);return e/(1.0+e)

def numeric_or_missing(value):
    if value is None or str(value).strip().lower() in {'','na','nan','none','null'}:return None
    x=float(value)
    if math.isnan(x):return None
    if not math.isfinite(x):raise ValueError('Infinite input values are not accepted.')
    return x

def predict_row(row,model,use_original_units=False):
    contributions=[]
    for record in model['features']:
        if record['feature'] not in row:raise ValueError('Missing required input column: '+record['feature'])
        value=numeric_or_missing(row.get(record['feature']))
        if value is None:value=record['imputation_median']
        if use_original_units:contributions.append(record['original_unit_coefficient']*value)
        else:contributions.append(record['standardized_coefficient']*(value-record['training_mean'])/record['training_scale'])
    intercept=model['original_unit_intercept'] if use_original_units else model['standardized_intercept']
    return expit_stable(math.fsum([intercept]+contributions))

def main():
    here=Path(__file__).resolve().parent;p=argparse.ArgumentParser()
    p.add_argument('--parameters',type=Path,default=here/'model_parameters.json')
    p.add_argument('--input',type=Path,default=here/'example_data.csv')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',help='Comma-separated saved model names; default all five.')
    p.add_argument('--original-units',action='store_true',help='Use equivalent original-unit coefficients instead of standardized coefficients.')
    args=p.parse_args();saved=json.loads(args.parameters.read_text(encoding='utf-8'));names=args.models.split(',') if args.models else list(saved['models'])
    unknown=set(names)-set(saved['models'])
    if unknown:raise ValueError('Unknown saved model name(s): '+', '.join(sorted(unknown)))
    with args.input.open(encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
    output=[]
    for row in rows:
        output.append({'node_id':row.get('node_id',''),'group_id':row.get('group_id',''),**{name+'_probability':format(predict_row(row,saved['models'][name],args.original_units),'.17g') for name in names}})
    with args.output.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(output[0]) if output else ['node_id','group_id']+[name+'_probability' for name in names]);w.writeheader();w.writerows(output)

if __name__=='__main__':main()
