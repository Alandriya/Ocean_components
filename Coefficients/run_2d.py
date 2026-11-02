"""Command-line entry point for aligned 2D NP and SP experiments.

Example:
    python run_2d.py --x0 sensible.npy --x1 latent.npy --mask ocean_mask.npy \\
        --out results --bins 12 --method both --time-end 15 --cf
"""
import argparse
from pathlib import Path
import numpy as np
from aligned_common import load_pair_series, load_water_mask, prepare_pair_transition
from nonparametric_aligned import count_2d_nonparam
from semiparametric_aligned import count_semi_2d_AB
from coefficients_cf import count_cf_from_daily


def main():
    parser = argparse.ArgumentParser(description='Aligned bivariate SDE coefficient estimation')
    parser.add_argument('--x0', required=True, help='First variable .npy')
    parser.add_argument('--x1', required=True, help='Second variable .npy')
    parser.add_argument('--mask', help='Binary water mask, 1=water, 0=land')
    parser.add_argument('--out', default='results')
    parser.add_argument('--bins', type=int, default=12, help='Quantile bins per axis (total up to bins^2)')
    parser.add_argument('--method', choices=('np', 'sp', 'both'), default='both')
    parser.add_argument('--time-start', type=int, default=0)
    parser.add_argument('--time-end', type=int, default=None)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--n-components', type=int, default=2)
    parser.add_argument('--n-init', type=int, default=10)
    parser.add_argument('--reg-covar', type=float, default=1e-6)
    parser.add_argument('--min-group-size', type=int, default=2)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--cf', action='store_true', help='Compute C, F, and FS from saved daily maps')
    parser.add_argument('--correlation-window', type=int, default=14)
    parser.add_argument('--mean-window', type=int, default=7)
    args = parser.parse_args()

    s0, s1 = load_pair_series(args.x0, args.x1)
    mask = load_water_mask(args.mask, s0.shape[1:])
    end = args.time_end if args.time_end is not None else len(s0)
    output_root = Path(args.out) / f'bins_{args.bins}'
    print(f'Input: {s0.shape}; water points: {mask.sum()}/{mask.size}; range: ({args.time_start}, {end})')
    print(f'Output root: {output_root}')
    first = prepare_pair_transition(s0[args.time_start], s0[args.time_start + 1],
                                    s1[args.time_start], s1[args.time_start + 1],
                                    args.bins, mask)
    available = sum(len(g.increments) >= args.min_group_size for g in first.groups)
    retained = sum(len(g.increments) for g in first.groups if len(g.increments) >= args.min_group_size)
    print(f'Joint cells: {first.n_occupied_bins}/{args.bins**2}; '
          f'usable groups: {available}; retained water observations: {retained}/{first.n_valid_pairs}')
    if retained == 0:
        raise ValueError('No groups of sufficient size; lower --bins or --min-group-size')
    for method in ('np', 'sp'):
        if args.method not in (method, 'both'):
            continue
        print(f'Estimating {method.upper()} ...', flush=True)
        if method == 'np':
            count_2d_nonparam(output_root, s0, s1, args.time_start, end,
                              path='Coefficients/', quantiles_amount=args.bins,
                              mask=mask, min_group_size=args.min_group_size,
                              overwrite=args.overwrite, dt=args.dt)
            method_folder = 'Nonparametric_2d'
        else:
            count_semi_2d_AB(output_root, s0, s1, args.time_start, end,
                             path='Coefficients/', quantiles_amount=args.bins,
                             mask=mask, min_group_size=args.min_group_size,
                             n_components=args.n_components, n_init=args.n_init,
                             reg_covar=args.reg_covar, overwrite=args.overwrite, dt=args.dt)
            method_folder = 'Semiparametric_2d'
        if args.cf:
            count_cf_from_daily(output_root, method_folder, args.time_start, end,
                                correlation_window=args.correlation_window,
                                mean_width=args.mean_window, mask=mask,
                                pair_name=method_folder)
    print('Finished.')


if __name__ == '__main__':
    main()
