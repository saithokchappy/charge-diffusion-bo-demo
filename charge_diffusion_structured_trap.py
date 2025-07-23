#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Charge Carrier Simulation with Structured Trap Density
=======================================================

This script simulates the diffusion, trapping, and recombination of charge carriers
on a 2D grid and uses Bayesian Optimization (Gaussian Process Regression) to find
the optimal initial distribution that maximizes the remaining carrier density.

Technologies used:
- NumPy: for 2D array computation and diffusion logic
- Matplotlib: for visualization and animation
- Scikit-learn: Gaussian Process Regression (GPR)
- SciPy: Acquisition function optimization using L-BFGS-B

Author: Yoshitaka Furuya

Author: Yoshitaka Furuya
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

def generate_initial_distribution(grid_size, x0, y0, sigma):
    """
    Make a 2D Gaussian centered at (x0, y0) with given spread.
    The sum is normalized to 1.
    """
    h, w = grid_size
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    gauss = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gauss / np.sum(gauss)

def create_structured_trap_mask(grid_size):
    """
    Create trap density as a smooth 2D function (sum of Gaussians).
    Convert that into a probabilistic mask.
    """
    h, w = grid_size
    y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')

    trap_density = (
        np.exp(-((x - 0.2)**2 + (y - 0.2)**2) / 0.01) +
        np.exp(-((x - 0.8)**2 + (y - 0.4)**2) / 0.005) +
        np.exp(-((x - 0.3)**2 + (y - 0.9)**2) / 0.01)
    )

    trap_density /= trap_density.max()
    trap_mask = np.random.rand(*grid_size) < trap_density * 0.5  # adjustable threshold
    return trap_mask, trap_density

def simulate_diffusion_trap_recombination(init_grid, trap_mask, D=0.1, k_recomb=0.01, steps=100, return_history=False):
    """
    Main simulation loop: diffusion + trap + recombination.
    Optionally returns history for animation.
    """
    grid = init_grid.copy()
    history = [grid.copy()] if return_history else None
    for _ in range(steps):
        lap = (
            np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) -
            4 * grid
        )
        grid += D * lap #diffusion
        grid[trap_mask] *= 0.9 # trap
        grid *= (1 - k_recomb) # recombination
        if return_history:
            history.append(grid.copy())
    return (grid, history) if return_history else grid

def evaluate(params, grid_size=(100, 100), trap_mask=None, D=0.1, k_recomb=0.01, steps=100):
    """
    Given (x0, y0, sigma), simulate and return -remaining charge.
    """
    x0, y0, sigma = params
    init = generate_initial_distribution(grid_size, x0, y0, sigma)
    final = simulate_diffusion_trap_recombination(init, trap_mask, D, k_recomb, steps)
    return -np.sum(final)

def create_animation(history, interval=100, save_path="diffusion.gif"):
    """
    Create a simple gif of the charge evolution over time.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(history[0], cmap='viridis', animated=True)
    def update(frame):
        im.set_array(history[frame])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=interval, blit=True)
    ani.save(save_path, writer='pillow')
    plt.close()
    print(f"[Saved] Animation: {save_path}")

def run_bayesian_optimization(bounds, n_init=5, n_iter=10):
    """
    Run BO using Gaussian Process + Lower Confidence Bound (LCB).
    Try to find initial distribution that preserves most charge.
    """
    np.random.seed(42)
    grid_size = (100, 100)
    trap_mask, trap_density = create_structured_trap_mask(grid_size)

    # visualize trap_density
    plt.imshow(trap_density, cmap='gray')
    plt.title("Structured Trap Density Map")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("trap_density_map.png")
    plt.show()

    X = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(n_init, len(bounds)))
    y = np.array([evaluate(p, grid_size, trap_mask) for p in X])
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    for i in range(n_iter):
        gpr.fit(X, y)
        def acquisition(x):
            x = np.array(x).reshape(1, -1)
            mu, sigma = gpr.predict(x, return_std=True)
            return -(mu - 1.96 * sigma)
        x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(acquisition, x0=x0, bounds=bounds, method='L-BFGS-B')
        x_new = res.x
        y_new = evaluate(x_new, grid_size, trap_mask)
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
        print(f"Iter {i+1}: params = {x_new.round(2)}, remaining = {-y_new:.4f}")

    best_idx = np.argmin(y)
    best_params = X[best_idx]
    print("\nBest Params [x0, y0, sigma] :", best_params)
    print("Max remaining charge:", -y[best_idx])
    best_init = generate_initial_distribution(grid_size, *best_params)
    best_final, history = simulate_diffusion_trap_recombination(best_init, trap_mask, return_history=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(best_init, cmap='viridis')
    axs[0].set_title("Best Init")
    axs[1].imshow(best_final, cmap='viridis')
    axs[1].set_title("Best Final")
    plt.tight_layout()
    plt.savefig("best_result.png")
    plt.show()

    create_animation(history, interval=100, save_path="diffusion.gif")

if __name__ == "__main__":
    bounds = [(20, 80), (20, 80), (2, 10)]
    run_bayesian_optimization(bounds)
