# Outbreak Detection

This repository contains a Python implementation of the outbreak detection algorithm described in the publication:

> Schlosser, F., Brockmann, D. (2021). Finding Disease Outbreak Locations from Human Mobility Data. (under review)

## Introduction

The outbreak detection method is intended to detect the location of _spatially localized_ disease outbreaks, meaning that individuals present at a certain location at a certain time are infected.

The outbreak method is given the spatial trajectory of the infected individuals, for example recorded from smartphone GPS, and aims to infer the outbreak origin from it.

The outbreak origin is estimated at the location where most individuals have been co-located at the same time at one point in the past.

<img src="/data/method_illustration.jpg" width="600">

## Installation

```shell
python -m pip install -e outbreak-detection
```

## Usage

Examples of how the outbreak detection can be used are given in the notebooks.

### Minimal Example

The following example shows a minimal application of the algorithm.

```python
import pandas as pd
from outbreak.util import load_sample_trajectories, plot_result

# Load trajectories of 4 individuals simulated using a dEPR model
trajectories = load_sample_trajectories()

# Load inference method
from outbreak import Inference
inference = Inference(GAUSS_SIGMA=1)

# Run inference on sample
origins = inference.find_outbreak_origins(trajectories)

# Plot results
from outbreak.util import plot_result
plot_result(trajectories, origins, n_origins=10)
```

<img src="/data/minimal_example.png" width="600">
