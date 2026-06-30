import datetime
import math
import os.path
from statistics import quantiles
from struct import unpack
import json
import numpy as np
from matplotlib.pyplot import autumn
from scipy.stats import mannwhitneyu
from skimage.metrics import structural_similarity as ssim
from torchgen.executorch.api.et_cpp import return_names

from Data_processing.data_processing import *
from Data_processing.data_processing import mean_blocks
from Data_processing.func_estimation import *
# from Plotting.plot_eigenvalues import plot_eigenvalues, plot_mean_year
# from Plotting.plot_extreme import *
# from extreme_evolution import *
# from ABCF_coeff_counting import *
from Eigenvalues.eigenvalues import *
from Plotting.plot_func_estimations import plot_ab_functional
# from Plotting.plot_Bel_coefficients import *
# from SRS_count_coefficients import *
# from Plotting.mean_year import *
from Plotting.video import *
from Coefficients.semiparametric import *
from Forecasting.utils import fix_random
from Plotting.plot_func_estimations import *
from Plotting.plot_coefficients import *
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import multipletests
from scipy.special import gammainc, gamma, erf, erfcx, gammaincc
# files_path_prefix = '/home/aosipova/EM_ocean/'
files_path_prefix = 'D:/Nastya/Data/OceanFull/'

width = 181
height = 161

block_size = 5
start_year = 1979
data1_name = 'sensible'
data2_name = 'latent'


str_types = f'{data1_name}-{data2_name}'
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
else:
    end_year = 2026
    coef_start = 14611
    coef_end = 17106

# start_year = 1979
# end_year = 2024

# ---------------------------------------------------------------------------------------
# Days deltas
days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
days_delta5 = (datetime.datetime(2025, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
days_delta6 = (datetime.datetime(2025, 11, 1, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
# ---------------------------------------------------------------------------------------

# days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
# days_delta7 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
days_delta8 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
# ----------------------------------------------------------------------------------------------

def get_mask():
    # Mask
    maskfile = open(files_path_prefix + "DATA/mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((height, width))
    return mask

def count_constants(params):
    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = params
    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    prefL = 2 * np.exp(-dL)
    prefC = 2 * np.exp(-dC)
    prefR = 2 * np.exp(-dR)

    const1 = -prefL * Phi(x1, c1, k)
    const2 = const1 + prefL * Phi(x1, c1, k) - prefC * Psi_minus(x1, c2, k)
    const3 = const2 + prefC * Psi_minus(x0, c2, k) - prefR * Psi_minus(x0, c4, k)
    const4 = const3 + prefR * Psi_minus(0, c4, k) - prefR * Psi_plus(0, c4, k)

    return [dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4]

# start_year = 19792024
if start_year == 1979:
    a1_rmse = 1.530e-01
    b1_rmse = 1.973e-01
    a2_rmse = 4.993e-01
    b2_rmse = 1.609e-01
    x0 = -3.647613525390625
    x1 = -40
    k4 = 1.026 * 10**(-7)
    k3 = 2.250 * 10**(-5)
    k2 = 5.919 * 10**(-4)
    k1 = -1.401 * 10 **(-1)
    k0 = 1.837 * 1
    k = [k0, k1, k2, k3, k4]
    x_min = -1118.5375366210938
    # x_max = 259.7684326171875
    x_max = 100

    c1 = 1.255 * 10**(-2)
    c2 = 3.367 * 10**(-1)
    c3 = 5.536
    c4 = 3.294 * 10**(-1)

    z = 1.343e-01
    sensible_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]

    x0 = -2.8423638983087227
    x1 = -40

    k4 = -1.230 * 10**(-9)
    k3 = 2.755 * 10**(-6)
    k2 = 1.230 * 10**(-3)
    k1 = 5.313 * 10 **(-2)
    k0 = -4.376 * 1
    k = [k0, k1, k2, k3, k4]

    c1 = 5.257 * 10**(-3)
    c2 = 1.951 * 10**(-1)
    c3 = 6.174
    c4 = 2.852 * 10**(-1)
    # x_min = -1225.4668579101562
    x_min = -300
    x_max = 225.44720458984375
    z = 1.271e-01
    latent_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]
elif start_year == 1989:
    a1_rmse = 4.723e-01
    a2_rmse = 1.002e+00
    b1_rmse = 1.930e-01
    b2_rmse = 2.978e-01

    x0 = -3.9940185546875
    x1 = -40

    k4 = -6.372e-07
    k3 = -1.435e-04
    k2 = -8.734e-03
    k1 = -3.072e-01
    k0 = -2.894e+00

    k = [k0, k1, k2, k3, k4]
    c1 = 1.531e-02
    c2 = 3.280e-01
    c3 = 5.577e+00
    c4 = 3.297e-01
    z = 7.466e-02
    x_min = -989.6334228515625
    x_max = 253.22821044921875
    # x_max = 300
    x_min  = -200
    sensible_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]

    x0 = -3.2288980482611276
    x1 = -100

    k4 = 1.498e-08
    k3 = 1.068e-05
    k2 = 3.273e-03
    k1 = 2.170e-01
    k0 = -5.413e+00
    k = [k0, k1, k2, k3, k4]

    c1 = 5.565e-03
    c2 = 2.244e-01
    c3 = 6.096e+00
    c4 = 3.460e-01
    x_min = -1303.3750610351562
    # x_min = -300
    # x_max = 251.348388671875
    x_max = 60
    z = 6.503e-02
    latent_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]
elif start_year == 1999:
    a1_rmse = 4.080e-01
    a2_rmse = 9.036e-01
    b1_rmse = 2.073e-01
    b2_rmse = 2.617e-01


    x0 = -1.44317626953125
    x1 = -40

    k4 = -6.163e-07
    k3 = -1.387e-04
    k2 = -8.406e-03
    k1 = -3.004e-01
    k0 = -2.966e+00

    k = [k0, k1, k2, k3, k4]
    c1 = 1.522e-02
    c2 = 3.505e-01
    c3 = 5.323e+00
    c4 = 4.354e-01
    z = 8.314e-02
    # x_min = -953.4976806640625
    x_min= -200
    x_max = 262.07489013671875
    sensible_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]

    x0 = -3.3470081595697354
    x1 = -40

    k4 = 1.690e-08
    k3 = 1.212e-05
    k2 = 3.538e-03
    k1 = 2.388e-01
    k0 = -5.270e+00
    k = [k0, k1, k2, k3, k4]

    c1 = 5.413e-03
    c2 = 2.210e-01
    c3 = 6.070e+00
    c4 = 3.186e-01
    x_min = -1304.8052978515625
    # x_max = 228.06671142578125
    x_max = 50
    z = 1.143e-01
    latent_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]
elif start_year == 2009:
    a1_rmse = 4.006e-01
    a2_rmse = 9.221e-01
    b1_rmse = 1.831e-01
    b2_rmse = 2.647e-01


    x0 = -4.130950106107264
    x1 = -40

    k4 = -6.713e-07
    k3 = -1.472e-04
    k2 = -8.728e-03
    k1 = -3.064e-01
    k0 = -2.972e+00

    k = [k0, k1, k2, k3, k4]
    c1 = 1.571e-02
    c2 = 3.110e-01
    c3 = 5.631e+00
    c4 = 3.052e-01
    z = 7.512e-02
    # x_min = -1059.0289916992188
    x_min = -200
    x_max = 232.53118896484375
    sensible_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]

    x0 = -4.089643541273176
    x1 = -40

    k4 = 1.096e-08
    k3 = 9.046e-06
    k2 = 3.028e-03
    k1 = 2.044e-01
    k0 = -5.896e+00
    k = [k0, k1, k2, k3, k4]

    c1 = 5.719e-03
    c2 = 2.053e-01
    c3 = 6.178e+00
    c4 = 2.838e-01
    x_min = -1427.2758178710938
    # x_max = 227.25
    x_max = 60
    z = 1.122e-01
    latent_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]
elif start_year == 19792024:
    print(start_year)
    a1_rmse = 4.225e-01
    a2_rmse = 7.880e-01
    b1_rmse = 2.280e-01
    b2_rmse = 2.567e-01


    x0 = -0.099578857421875
    x1 = -40

    k4 = -3.560e-07
    k3 = -7.899e-05
    k2 = -4.698e-03
    k1 = -2.318e-01
    k0 = -1.882e+00

    k = [k0, k1, k2, k3, k4]
    c1 = 1.733e-02
    c2 = 4.263e-01
    c3 = 4.402e+00
    c4 = 5.151e-01
    z = 1
    x_min = -1118.5375366210938
    # x_min = -200
    x_max = 262.07489013671875
    sensible_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]

    x0 = -1.3424892327724312
    x1 = -40

    k4 = 1.325e-08
    k3 = 1.025e-05
    k2 = 3.153e-03
    k1 = 2.324e-01
    k0 = -2.308e+00
    k = [k0, k1, k2, k3, k4]

    c1 = 6.180e-03
    c2 = 3.315e-01
    c3 = 4.843e+00
    c4 = 5.134e-01
    x_min = -1427.2758178710938
    # x_min = -300
    x_max = 251.348388671875
    z = 1.866e-01
    latent_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]
else:
    a1_rmse = 3.730e+01
    a2_rmse = 1.036e+02
    b1_rmse = 2.284e+00
    b2_rmse = 1.998e+00


    x0 = -0.18399552565354563
    x1 = -40

    k4 = -7.235e-06
    k3 = -1.476e-03
    k2 = -9.780e-02
    k1 = 5.945e-01
    k0 = -9.889e+00

    k = [k0, k1, k2, k3, k4]
    c1 = 1.860e-02
    c2 = 7.282e-01
    c3 = 1.261e+01
    c4 = 7.451e-01
    z = 1.944e-05
    x_min = -951.7299194335938
    x_max = 252.967529296875
    sensible_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]

    x0 = -0.3771931015964706
    x1 = -40

    k4 = 4.965e-07
    k3 = 2.815e-04
    k2 = 5.692e-02
    k1 = 5.656e+00
    k0 = -3.231e-01
    k = [k0, k1, k2, k3, k4]

    c1 = 1.369e-02
    c2 = 2.078e-01
    c3 = 1.491e+01
    c4 = 5.641e-01
    x_min = -1295.7306518554688
    x_max = 235.8623046875
    z = 1.086e-05
    latent_params = [x0, x1, k, x_min, x_max, c1, c2, c3, c4, z]

"""
A1 rmse:  2.471e-01
-9.475e-08 * x^4 + -2.825e-05 * x^3 + -2.147e-03 * x^2 +-2.449e-01 * x + -2.108e+00
x0 = -1.6493380957753576
C3 = 4.703e+00
B11 RMSE: 2.261e-01
1.742e-02 * |x| + 6.062e+00
3.251e-01 * sqrt(|x|)  + 4.703e+00
3.828e-01 * sqrt(|x|) + 4.629e+00



A2 rmse:  2.438e+00
-1.289e-08 * x^4 + -6.425e-06 * x^3 + -2.456e-04 * x^2 +-1.253e-02 * x + -6.411e+00
x0 = -0.9844424067258247
C3 = 4.942e+00
4.650e-03 * |x| + 6.920e+00
3.126e-01 * sqrt(|x|)  + 4.942e+00
5.036e-01 * sqrt(|x|) + 4.752e+00
B22 RMSE: 3.400e-01
"""