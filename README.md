# Ocean_components
 
## Description
This repository contains python scripts for parsing, plotting, estimating and predicting oceanic data 
of several variables (sensible and latent heat fluxes, their sum, sea surface temperature - SST and atmospheric pressure)
in the area of North Atlantic, implementing the It'o stochastic differential equation (SDE) model:

$$
    dX = a(t,X)\,dt + b(t, X)\,dW(t), 
$$

where $X(t)$ is a random process whose values represent the values of the observed variables,
$a(t,X)$ is a drift coefficient and $b(t,X)$ is a diffusion coefficient, $dW(t)$ are the increments of the Wiener 
process (a Gaussian white noise).

There are 5 folders in the repository, each contaning scripts for a part of the work:
- **Coefficients** folder contains the scripts for estimating random coefficients $a(t,X)$ and $b(t,X)$, given the values of 
the process $X(t)$;
- **Data_processing** folder contains the scripts for parsing .grib files raw data from the ERA-5 database 
(https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), also the basic analysing of the data 
(estimating extreme values, retrieving trends evolution, estimating the distributions, etc.);
- **Eigenvalues** folder contains the script for constructing Karhunen-Lo'eve decomposition of the diffusion coefficient;
- **Forecasting** folder contains the scripts for building several type of models for creating mid-term forecasts of the
Flux variable (considering 3, 5, 7 and 10 days length horizons);
- **Plotting** folder contains scripts for plotting variables, coefficients, eigenvectors, extreme trends, forecasts and other;
- **Video examples** folder contains several examples of the short videos of the coefficients estimations and eigenvectors, 
which could be created with this software.

## Installation from github
If you want to use this code, you should follow the following steps.

### 1. Download 
Download the source code from the Github repository: https://github.com/Alandriya/Ocean_components

### 2. Create the virtual environment
You should create a virtual environment in the same direction where you have the folder of this project, e.g.:

C:/Python projects/Venv

C:/Python projects/Ocean_components

After that you should install the required libraries, specified in the file requirements.txt.

### 3. Launch
If you want to **build or use the forecasting models with neural networks**, you should launch the nn_train.py or nn_test.py scripts from the 
command prompt via commands:

..\Venv\Scripts\activate

torchrun --nproc_per_node=1 --master_port 39985 nn_train.py

You can change the master_port or the nproc_per_node arguments, according to your needs. 

Forecasting scripts with **other ML models** can be launched from your IDE or from the command prompt with the script 
ml_runner.py

If you want to use the script for **processing .grib files** into the numpy arrays, you should launch the grib_handler.py 
script from your IDE or from the command prompt.

For **all other aims** (constructing the random coefficients estimations or the eigenvectors and eigenvalues from the 
Karhunen-Lo'eve decomposition of the diffusion coefficient, getting the extreme trends, etc.) you should launch 
the main.py script, uncommenting the needed sections. Some old sections of the main.py script can be found in the 
Data_processing/main_parts.py script.

## Visual representation
The considered North Atlantic region is represented as a grid which boundaries in latitude range from $-90$ to $0$ 
degrees, and in longitude from $0$ to $80$ degrees. Here you can see some of the maps with the examples of the created 
plots:

![](video examples\16436.png)
<center>An example of the data used: total heat flux, SST and surface pressure, averaged data for January, $1$, 
$2024$.
</center>

![](video examples\press-press_eigenvectors.png)
<center>The first three eigenvectors from the ordered set for the Pressure-Pressure pair on 
$01.01.23$, $01.04.23$, $01.07.23$ and $01.10.23$
</center>

## Related publications
If you use this repository in your work, please consider citing one or more of these publications:

- Gorshenin A. K., Osipova A. A., Belyaev K. P. Stochastic analysis of air-sea heat fluxes variability in the North Atlantic in 1979–2022 based on reanalysis data // Computers and Geosciences, 2023. Vol. 181. Art. No. 105461. DOI: 10.1016/j.cageo.2023.105461
- Belyaev K., Gorshenin A., Korolev V., Osipova A. Comparison of statistical approaches for reconstructing random coefficients in the problem of stochastic modeling of air–sea heat flux increments / // Mathematics, 2024. Vol. 12, no. 2. Art. No. 288. DOI: 10.3390/math12020288 