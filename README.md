# Outbreak Detection

This repository contains a Python implementation of the outbreak detection algorithm described in the publication:

> Schlosser, F., Brockmann, D. (2021). Finding Disease Outbreak Locations from Human Mobility Data. (under review)

## Introduction

The outbreak detection method is intended to detect the location of _spatially localized_ disease outbreaks. In this scenario, individuals that were present at a certain location at a certain time were potentially exposed to the infection.

The outbreak method is given the spatial trajectory of a sample of infected individuals, for example recorded from smartphone GPS, and aims to infer the outbreak origin from it.

The outbreak origin is estimated at the location where most individuals have been co-located at the same time at one point in the past.

<img src="/data/method_illustration.jpg" width="600">

Although the method is motivated by infectious diseases, the algorithm can be used in other contexts to detect co-locations of individuals in spatial trajectories.

## Installation

Clone the repository and run

```shell
python -m pip install -e outbreak-detection
```

## Usage

Examples of how the outbreak detection can be used are given in the notebooks.

### Minimal Example

The following example shows a minimal application of the algorithm.

```python
from outbreak import Inference
from outbreak.util import load_sample_trajectories, plot_result

# Load trajectories of 4 individuals simulated using a dEPR model
trajectories = load_sample_trajectories()

# Instantiate the inference method
inference = Inference(GAUSS_SIGMA=1)

# Run inference on the sample
origins = inference.find_outbreak_origins(trajectories)

# Plot results
plot_result(trajectories, origins, n_origins=10)
```

<img src="/data/minimal_example.png" width="600">
