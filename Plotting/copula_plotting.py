import os
import numpy as np
import matplotlib.pyplot as plt

def plot_copula_marginals(files_path_prefix, x1, p1, marginal_x1, x2, p2, marginal_x2):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x1, p1, label="1D stationary")
    ax.plot(x1, marginal_x1, "--", label="2D copula marginal")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(files_path_prefix + f'videos/Functional/2d_copula_sensible_marginal.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x2, p2, label="1D stationary")
    ax.plot(x2, marginal_x2, "--", label="2D copula marginal")
    ax.set_xlabel("Latent heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(files_path_prefix + f'videos/Functional/2d_copula_latent_marginal.png')
    plt.close(fig)


def plot_joint_density(files_path_prefix, x1, x2, joint_density, title="Gaussian copula stationary density", xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x1, x2, joint_density, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Stationary density")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/2d_density.png')
    plt.close(fig)


def plot_joint_contours(files_path_prefix, x1, x2, joint_density, levels=15, title="Gaussian copula stationary density", xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    contour = ax.contour(x1, x2, joint_density, levels=levels)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/joint_contours.png')
    plt.close(fig)


def plot_empirical_copula_comparison(files_path_prefix, x1, x2, empirical_density, copula_density, xlim=None, ylim=None, same_scale=True):
    vmax = max(np.nanmax(empirical_density), np.nanmax(copula_density)) if same_scale else None

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x1, x2, empirical_density, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="Density")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Empirical joint density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/empirical_joint_density.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x1, x2, copula_density, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="Density")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Gaussian copula stationary density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/copula_density_2d.png')
    plt.close(fig)


def plot_empirical_copula_contours(files_path_prefix, x1, x2, empirical_density, copula_density, levels=12, xlim=None, ylim=None):
    empirical_scaled = empirical_density / np.nanmax(empirical_density)
    copula_scaled = copula_density / np.nanmax(copula_density)
    contour_levels = np.linspace(0.05, 0.95, levels)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.contour(x1, x2, empirical_scaled, levels=contour_levels, linewidths=1.5)
    ax.contour(x1, x2, copula_scaled, levels=contour_levels, linestyles="--", linewidths=1.5)
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Empirical (solid) vs Gaussian copula (dashed)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(files_path_prefix + f'videos/Functional/joint_contours_empirical.png')
    plt.close(fig)

def plot_copula_space_comparison(files_path_prefix, centers, empirical_density, gaussian_density, levels=12):
    vmax = max(np.nanmax(empirical_density), np.nanmax(gaussian_density))

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(centers, centers, empirical_density, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="Copula density")
    ax.set_xlabel("U = F1(X1)")
    ax.set_ylabel("V = F2(X2)")
    ax.set_title("Empirical copula")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(files_path_prefix + f'videos/Functional/copula_empirical.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(centers, centers, gaussian_density, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="Copula density")
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_title("Gaussian copula")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(files_path_prefix + f'videos/Functional/copula_gaussian.png')
    plt.close(fig)


def plot_copula_space_contours(files_path_prefix, centers, empirical_density, gaussian_density, levels=12):
    empirical_scaled = empirical_density / np.nanmax(empirical_density)
    gaussian_scaled = gaussian_density / np.nanmax(gaussian_density)
    contour_levels = np.linspace(0.05, 0.95, levels)

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.contour(centers, centers, empirical_scaled, levels=contour_levels, linewidths=1.5)
    ax.contour(centers, centers, gaussian_scaled, levels=contour_levels, linestyles="--", linewidths=1.5)
    ax.plot([0, 1], [0, 1], ":", linewidth=1)
    ax.set_xlabel("U = F1(X1)")
    ax.set_ylabel("V = F2(X2)")
    ax.set_title("Empirical copula (solid) vs Gaussian copula (dashed)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(files_path_prefix + f'videos/Functional/copula_space_contours.png')
    plt.close(fig)

def plot_copula_models_contours(files_path_prefix, centers, empirical_density, gaussian_density, student_density, levels=10):
    empirical_scaled = empirical_density / np.nanmax(empirical_density)
    gaussian_scaled = gaussian_density / np.nanmax(gaussian_density)
    student_scaled = student_density / np.nanmax(student_density)
    contour_levels = np.linspace(0.05, 0.9, levels)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    ax.contour(centers, centers, empirical_scaled, levels=contour_levels, colors="black", linewidths=1.5)
    ax.contour(centers, centers, gaussian_scaled, levels=contour_levels, colors="orange", linestyles="--", linewidths=1.2)
    ax.contour(centers, centers, student_scaled, levels=contour_levels, colors="green", linestyles=":", linewidths=1.5)

    ax.plot([], [], linestyle="-", label="Empirical", c='black')
    ax.plot([], [], linestyle="--", label="Gaussian", c='orange')
    ax.plot([], [], linestyle=":", label="Student t", c='g')

    ax.set_xlabel("U = F1(X1)")
    ax.set_ylabel("V = F2(X2)")
    ax.set_title("Empirical vs Gaussian vs Student t copula")
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0.01, 0.99)
    ax.legend()

    fig.savefig(files_path_prefix + "videos/Functional/copula_models_contours.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_copula_density(files_path_prefix, centers, density, filename, title):
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)

    mesh = ax.pcolormesh(centers, centers, density, shading="auto", cmap="YlOrRd")
    fig.colorbar(mesh, ax=ax, label="Copula density")

    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_title(title)
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0.01, 0.99)

    fig.savefig(files_path_prefix + f"videos/Functional/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_figure(files_path_prefix, fig, filename):
    os.makedirs(files_path_prefix + "videos/Functional/", exist_ok=True)
    fig.savefig(files_path_prefix + f"videos/Functional/{filename}", dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_checkerboard_copula(files_path_prefix, centers, density, xlim=(0.02, 0.98), ylim=(0.02, 0.98)):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = ax.pcolormesh(centers, centers, density, shading="auto", cmap="YlOrRd")
    fig.colorbar(mesh, ax=ax, label="Copula density")

    ax.set_xlabel("U = F1(X1)")
    ax.set_ylabel("V = F2(X2)")
    ax.set_title("Empirical checkerboard copula")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    save_figure(files_path_prefix, fig, "copula_checkerboard.png")


def plot_checkerboard_stationary_density(files_path_prefix, x1, x2, joint_density, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x1, x2, joint_density, shading="auto", cmap="YlOrRd")
    fig.colorbar(mesh, ax=ax, label="Stationary density")

    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Checkerboard copula stationary density")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    save_figure(files_path_prefix, fig, "checkerboard_stationary_density.png")


def plot_checkerboard_marginals(files_path_prefix, x1, p1, marginal_x1, x2, p2, marginal_x2):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x1, p1, label="1D stationary")
    ax.plot(x1, marginal_x1, "--", label="Checkerboard 2D marginal")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(files_path_prefix, fig, "checkerboard_marginal_x1.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x2, p2, label="1D stationary")
    ax.plot(x2, marginal_x2, "--", label="Checkerboard 2D marginal")
    ax.set_xlabel("Latent heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(files_path_prefix, fig, "checkerboard_marginal_x2.png")


def plot_empirical_checkerboard_contours(files_path_prefix, x1, x2, empirical_density, checkerboard_density, levels=10, xlim=None, ylim=None):
    empirical_scaled = empirical_density / np.nanmax(empirical_density)
    checkerboard_scaled = checkerboard_density / np.nanmax(checkerboard_density)
    contour_levels = np.linspace(0.05, 0.9, levels)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.contour(x1, x2, empirical_scaled, levels=contour_levels, linewidths=1.5)
    ax.contour(x1, x2, checkerboard_scaled, levels=contour_levels, linestyles="--", linewidths=1.5)

    ax.plot([], [], "-", label="Empirical")
    ax.plot([], [], "--", label="Checkerboard stationary")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Empirical vs checkerboard stationary density")
    ax.legend()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    save_figure(files_path_prefix, fig, "empirical_checkerboard_contours.png")

def plot_bernstein_copula(files_path_prefix, grid, density):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = ax.pcolormesh(grid, grid, density, shading="auto", cmap="YlOrRd")
    fig.colorbar(mesh, ax=ax, label="Copula density")
    ax.set_xlabel("U = F1(X1)")
    ax.set_ylabel("V = F2(X2)")
    ax.set_title("Bernstein copula")
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0.01, 0.99)

    fig.savefig(files_path_prefix + "videos/Functional/copula_bernstein.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bernstein_stationary_density(files_path_prefix, x1, x2, density, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x1, x2, density, shading="auto", cmap="YlOrRd")
    fig.colorbar(mesh, ax=ax, label="Stationary density")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Latent heat flux")
    ax.set_title("Bernstein copula stationary density")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.savefig(files_path_prefix + "videos/Functional/bernstein_stationary_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bernstein_marginals(files_path_prefix, x1, p1, marginal_x1, x2, p2, marginal_x2):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x1, p1, label="1D stationary")
    ax.plot(x1, marginal_x1, "--", label="Bernstein 2D marginal")
    ax.set_xlabel("Sensible heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(files_path_prefix + "videos/Functional/bernstein_marginal_x1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x2, p2, label="1D stationary")
    ax.plot(x2, marginal_x2, "--", label="Bernstein 2D marginal")
    ax.set_xlabel("Latent heat flux")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(files_path_prefix + "videos/Functional/bernstein_marginal_x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_checkerboard_bernstein_comparison(files_path_prefix, checkerboard, bernstein, levels=10):
    checker_centers = checkerboard["centers"]
    checker_density = checkerboard["density"]
    grid = bernstein["grid"]
    bernstein_density = bernstein["density"]

    checker_scaled = checker_density / np.max(checker_density)
    bernstein_scaled = bernstein_density / np.max(bernstein_density)
    contour_levels = np.linspace(0.05, 0.9, levels)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.contour(checker_centers, checker_centers, checker_scaled, levels=contour_levels, colors="black", linewidths=1.5)
    ax.contour(grid, grid, bernstein_scaled, levels=contour_levels, colors="red", linestyles="--", linewidths=1.5)

    ax.plot([], [], "-", label="Checkerboard", c='black')
    ax.plot([], [], "--", label="Bernstein", c='red')
    ax.set_xlabel("U = F1(X1)")
    ax.set_ylabel("V = F2(X2)")
    ax.set_title("Checkerboard vs Bernstein copula")
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0.01, 0.99)
    ax.legend()

    fig.savefig(files_path_prefix + "videos/Functional/checkerboard_bernstein_contours.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bernstein_m_sensitivity(files_path_prefix):
    m_values = np.array([30, 40, 50, 60, 70])

    correlation = np.array([
        0.48003290516952485, 0.4930341404032566, 0.5011049214969485,
         0.507138952874245, 0.5111178229066088]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(m_values, correlation, marker='o', linewidth=2.5)
    ax.axvline(50, linestyle='--', linewidth=1.5, label='Selected m = 50')

    ax.set_xlabel('Bernstein degree m')
    ax.set_ylabel('Pearson correlation')
    ax.set_xticks(m_values)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(files_path_prefix + 'videos/Functional/copula_m_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)