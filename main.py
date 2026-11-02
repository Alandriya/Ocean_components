from main_parts import (
    run_distribution_workflow,
    run_eigenvalues_for_period,
    run_historical_eigenvalues,
    read_ocean_mask,
)
from Plotting.plot_eigenvalues import plot_eigenvalues
from Data_processing.data_processing import scale_to_bins
from datetime import date, timedelta
import numpy as np
from Coefficients.aligned_common import load_series, load_water_mask
from Coefficients.runner import run_decade_coefficients
from Coefficients.diagnose_joint_groups import diagnose_joint_groups


if __name__ == "__main__":
    # Folder containing input data and calculated results.
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    mask = load_water_mask(read_ocean_mask(files_path_prefix))
    n_bins = 250
    n_components = 3
    n_init = 10
    run_1d = False  # Disable scalar calculations: only the paired process is requested.
    run_2d = True  # Run both NP and SP for the paired state (sensible, latent).
    n_bins_2d = 12  # 12 quantiles per coordinate: up to 144 joint groups.
    diagnostic_only = False  # First inspect group sizes; do NOT fit GMM over the decade.
    diagnostic_days = 7
    chunk_transitions = 7  # Progress reporting and resumable calculation.
    dt = 1.0
    overwrite = False
    names = ('sensible', 'latent')


    start_year = 1989
    end_year = start_year + 10
    if start_year == 1979:
        coef_start = 1
        coef_end = 3653
    elif start_year == 1989:
        coef_start = 3654
        coef_end = 7305
    elif start_year == 1999:
        coef_start = 7306
        coef_end = 10958
    elif start_year == 2009:
        coef_start = 10959
        coef_end = 14610
    elif start_year == 2019:
        end_year = 2026
        coef_start = 14611
        coef_end = 17106
    else:
        start_year = 1979
        coef_start = 1
        end_year = 2024
        coef_end = 17106
    offset = coef_start

    dir1 = dir2 = 'Fluxes'
    if names[0] == 'SST':
        dir1 = 'SST'
    elif names[0] == 'Pressure':
        dir1 = 'Pressure'

    if names[1] == 'SST':
        dir2 = 'SST'
    elif names[1] == 'Pressure':
        dir2 = 'Pressure'

    array1 = load_series(files_path_prefix + f'DATA/{dir1}/{names[0]}_{start_year}-{end_year}_grouped.npy')
    array2 = load_series(files_path_prefix + f'DATA/{dir2}/{names[1]}_{start_year}-{end_year}_grouped.npy')

    if diagnostic_only:
        # Use the exact same 2D grouping function as both coefficient estimators.
        # Test the first seven transitions (labels 2..8 when offset=1).
        diagnose_joint_groups(
            array1=array1,
            array2=array2,
            mask=mask,
            out_dir=(
                files_path_prefix
                + f"Coefficients/aligned_{start_year}-{end_year}/group_diagnostics"
            ),
            bins_list=(10, 12, 15, 18, 20),
            n_transitions=diagnostic_days,
            time_start=0,
            index_offset=offset,
            min_group_size=2,
            thresholds=(2, 10, 20, 30),
            plot_bins=15,
            make_plots=True,
        )
    else:
        # Switch diagnostic_only to False to calculate the full decade.
        run_decade_coefficients(
            files_path_prefix=files_path_prefix,
            array1=array1,
            array2=array2,
            mask=mask,
            names=names,
            start_year=start_year,
            end_year=end_year,
            start_index=offset,
            n_bins=n_bins,
            n_components=n_components,
            n_init=n_init,
            dt=dt,
            overwrite=overwrite,
            chunk_transitions=chunk_transitions,
            run_1d=run_1d,
            run_2d=run_2d,
            n_bins_2d=n_bins_2d,
        )

    # Intentionally stop here; this is the original main workflow switch.
    raise ValueError



    save_counts = False
    run_eigenvalues_for_period(
        files_path_prefix=files_path_prefix,
        start_year=start_year,
        end_year=end_year,
        n_bins=n_bins,
        n_components=n_components,
        dt=dt,
        overwrite=overwrite,
        save_counts=save_counts,
    )

    # Distribution estimation settings (task = "distributions").
    b_type = "simple"
    include_isolines = False
    include_copulas = False
    copula_bins = (50,)

    names = ("sensible", "latent")
    for names in [("sensible", "latent"), ('Flux', 'Flux'), ('SST', 'SST'), ('Pressure', 'Pressure'),
                  ('Flux', 'SST'), ('SST', 'Pressure'), ('Flux', 'Pressure')]:

        dir1 = dir2 = 'Fluxes'
        if names[0] == 'SST':
            dir1 = 'SST'
        elif names[0] == 'Pressure':
            dir1 = 'Pressure'

        if names[1] == 'SST':
            dir2 = 'SST'
        elif names[1] == 'Pressure':
            dir2 = 'Pressure'

        array1 = np.load(files_path_prefix + f'DATA/{dir1}/{names[0]}_{start_year}-{end_year}_grouped.npy')
        array2 = np.load(files_path_prefix + f'DATA/{dir2}/{names[1]}_{start_year}-{end_year}_grouped.npy')
        scaled1, quantiles1 = scale_to_bins(array1)
        scaled2, quantiles2 = scale_to_bins(array2)

        plot_eigenvalues(
            files_path_prefix=files_path_prefix,
            n_lambdas=3,
            mask=read_ocean_mask(files_path_prefix),
            t_start=0,
            t_end=5,
            offset=offset,
            array1=array1,
            array1_quantiles=quantiles1,
            array2=array2,
            array2_quantiles=quantiles2,
            names=names,
            shape=(161, 181),
        )





    # run_historical_eigenvalues(
    #     files_path_prefix=files_path_prefix,
    #     n_bins=n_bins,
    #     n_components=n_components,
    #     dt=dt,
    #     overwrite=overwrite,
    #     save_counts=save_counts,
    # )


    # run_distribution_workflow(
    #     files_path_prefix=files_path_prefix,
    #     start_year=start_year,
    #     end_year=end_year,
    #     b_type=b_type,
    #     include_isolines=include_isolines,
    #     include_copulas=include_copulas,
    #     copula_bins=copula_bins,
    # )

