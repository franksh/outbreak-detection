# Outbreak Detection

## Introduction

<!-- ![Inference Method Illustration](/data/method_illustration.png) -->
<img src="/data/method_illustration.png" width="600">

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
