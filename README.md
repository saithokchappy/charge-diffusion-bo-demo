# Charge Diffusion Simulation with Bayesian Optimization

This project simulates 2D charge carrier diffusion with trapping and recombination effects,
and uses Bayesian Optimization to find optimal initial distribution parameters
that maximize the remaining charge.

The trap regions are structured (3 Gaussian "hills"), so that the optimization path
can be visually and statistically understood.

## Features

- Diffusion model with discrete Laplacian
- Spatial traps and temporal recombination
- Structured trap landscape for interpretability
- Gaussian Process Regression + Lower Confidence Bound optimization
- Visualization of results and evolution animation

## How to Run

```bash
python charge_diffusion_structured_trap.py
```

This will output:

- trap_density_map.png

- best_result.png

- diffusion.gif

## Requirements

- Python 3.8+

- numpy

- matplotlib

- scipy

- scikit-learn

- pillow (for GIF export)

You can install dependencies with:
```bash
pip install -r requirements.txt
```

## Author
Yoshitaka Furuya