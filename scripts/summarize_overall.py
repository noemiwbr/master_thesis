"""Pool observation-level evaluation statistics across bidding zones."""
from pathlib import Path
import numpy as np
import pandas as pd

folder = Path('results/pooled')
rows = []
for model in ['median', 'shifted', 'BNN', 'QRNN']:
    for dataset, split in [('nowcast', 'train'), ('nowcast', 'validation'),
                           ('forecast', 'validation'), ('forecast', 'test')]:
        paths = [folder / f'{model}_{dataset}_{split}_NO{i}.npz' for i in range(1, 5)]
        if not all(p.exists() for p in paths):
            continue  # Never report a partial four-zone result as overall.
        data = [np.load(p) for p in paths]
        mse = np.concatenate([d['squared_errors'].reshape(-1) for d in data])
        crps = np.concatenate([d['crps_values'].reshape(-1) for d in data])
        mace = np.nan
        if model in ('BNN', 'QRNN'):
            levels = data[0]['quantile_levels']
            if not all(np.array_equal(d['quantile_levels'], levels) for d in data):
                raise ValueError('Quantile grids differ across zones')
            counts = sum((d['coverage_counts'] for d in data))
            n = sum(int(d['n_observations']) for d in data)
            mace = round(float(np.mean(np.abs(counts / n - levels))), 4)
        rows.append(dict(model=model, dataset=dataset, split=split,
                         bidding_zone='overall', rmse=round(float(np.sqrt(mse.mean())), 2),
                         crps=round(float(crps.mean()), 2), mace=mace))

if rows:
    result = pd.DataFrame(rows).sort_values(['model', 'dataset', 'split'])
    out = Path('results/overall_evaluation_results.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print('\nPooled results across NO1–NO4:\n', result.to_string(index=False))
else:
    print('No complete four-zone evaluation groups found.')
