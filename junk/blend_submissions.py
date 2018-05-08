from argparse import ArgumentParser

import numpy as np
import pandas as pd

from scipy.special import logit, expit


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--submissions', nargs='+')
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--mix-logits', action='store_true')
    parser.add_argument('--output-file')
    args = parser.parse_args()

    n = 18790469
    blend = np.zeros(n, dtype=np.float32)
    wnorm = sum(args.weights)
    for j, fname in enumerate(args.submissions):
        df = pd.read_csv(fname)
        df.sort_values(by=['click_id'], inplace=True)
        values = df['is_attributed'].values
        if args.mix_logits:
            values = logit(values)
        blend += args.weights[j] * values / wnorm

    if args.mix_logits:
        blend = expit(blend)

    df_out = pd.DataFrame(data={'click_id': np.arange(n, dtype=np.int32), 'is_attributed': blend})
    df_out.to_csv(args.output_file, header=True, index=False)
