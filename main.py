from main_parts import (
    run_distribution_workflow,
    run_eigenvalues_for_period,
    run_historical_eigenvalues,
)


if __name__ == "__main__":
    # Папка с исходными данными и расчётными результатами.
    files_path_prefix = "D:/Nastya/Data/OceanFull/"

    # Выбор сценария:
    # "eigenvalues"            — один период, все 10 пар из 4 переменных;
    # "historical-eigenvalues" — все четыре периода с 1979 по 2019 год;
    # "distributions"          — стационарные распределения и (опц.) копулы.
    task = "eigenvalues"

    # Сначала 2019—последний доступный день в файлах 2019-2026.
    # Для следующих запусков задавать, например:
    # start_year, end_year = 1979, 1989
    # start_year, end_year = 1989, 1999
    # start_year, end_year = 1999, 2009
    # start_year, end_year = 2009, 2019
    start_year = 2019
    end_year = 2026

    # Настройки вычисления eigenvalues.
    n_bins = 100
    n_components = 3
    dt = 1.0
    overwrite = True       # True — пересчитать существующие файлы.
    save_counts = False    # True — сохранять количество точек в каждой ячейке.

    # Настройки оценки распределений (task = "distributions").
    b_type = "simple"
    include_isolines = False
    include_copulas = False
    copula_bins = (50,)

    if task == "eigenvalues":
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

    elif task == "historical-eigenvalues":
        run_historical_eigenvalues(
            files_path_prefix=files_path_prefix,
            n_bins=n_bins,
            n_components=n_components,
            dt=dt,
            overwrite=overwrite,
            save_counts=save_counts,
        )

    elif task == "distributions":
        run_distribution_workflow(
            files_path_prefix=files_path_prefix,
            start_year=start_year,
            end_year=end_year,
            b_type=b_type,
            include_isolines=include_isolines,
            include_copulas=include_copulas,
            copula_bins=copula_bins,
        )

    else:
        raise ValueError(f"Неизвестный сценарий: {task}")
