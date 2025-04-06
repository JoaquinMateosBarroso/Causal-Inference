# Group Causation

Group Causation is an open source Python library for causal discovery on time series and groups of time series. It provides tools for identifying causal relationships, automating benchmarks, and evaluating causal inference methods. The library is designed for researchers and practitioners in causal inference, machine learning, and time series analysis. Check the [docs](https://joaquinmateosbarroso.github.io/group-causation/) for more detailed information.

![Untitled](https://github.com/user-attachments/assets/25aa8679-4185-4d1a-808e-5527c80a301d)


## Features
- **Causal Discovery**: Implements various causal discovery algorithms tailored for time series and grouped time series.
- **Benchmark Automation**: Automates benchmarking of causal inference methods across multiple datasets and configurations.
- **Evaluation Metrics**: Provides standardized evaluation metrics for assessing causal discovery performance.
- **Dataset Handling**: Supports synthetic and real-world datasets with preprocessing utilities.

## Installation

This library has been tested using Python 3.9, and it is recommended to install it in this version. You can install the library via pip:
```sh
pip install git+https://github.com/JoaquinMateosBarroso/group-causation
```

Or install from source for development purposes:
```sh
git clone https://github.com/JoaquinMateosBarroso/group-causation
cd group-causation
pip install -e .
```

## Usage

### Example: Running a Causal Discovery Algorithm

```python
import pandas as pd
import matplotlib.pyplot as plt
from group_causation.causal_discovery import PCMCIWrapper
from group_causation.create_toy_datasets import plot_ts_graph

data = pd.read_csv('your_dataset.csv').values
pcmci = PCMCIWrapper(data)
parents = pcmci.extract_parents()
plot_ts_graph(parents)
plt.show()
```

### Example: Running Benchmarks

```python
from group_causation.benchmark import BenchmarkCausalDiscovery
from group_causation.functions_test_data import static_parameters
from group_causation.causal_discovery import PCMCIWrapper, GrangerWrapper

algorithms = {'pcmci': PCMCIWrapper, 'granger': GrangerWrapper}
dataset_options = {
    'crosslinks_density': 0.5, # Portion of links that won't be in the kind of X_{t-1}->X_t; between 0 and 1
    'T': 2000, # Number of time points in the dataset
    'N_vars': 10, # Number of variables in the dataset
    'confounders_density': 0.2, # Portion of dataset that will be overgenerated as confounders; between 0 and inf
    'dependency_coeffs': [-0.3, 0.3], # default: [-0.5, 0.5]
    'auto_coeffs': [0.5], # default: [0.5, 0.7]
    'noise_dists': ['gaussian'], # deafult: ['gaussian']
    'noise_sigmas': [0.3], # default: [0.5, 2]
}    

# default, static parameters
parameters_iterator = static_parameters(dataset_options, {'pcmci': {}, 'granger': {}})

benchmark = BenchmarkCausalDiscovery()
benchmark.benchmark_causal_discovery(algorithms, parameters_iterator, generate_toy_data=True,
                                     datasets_folder='toy_data', results_folder='results', verbose=1)
benchmark.plot_particular_result('results')
```

## Supported Time Series Causal Discovery Methods
- PC-Stable
- PCMCI
- LPCMCI
- Granger Causality
- VARLiNGAM
- DYNOTEARS

## Supported Group Time Series Causal Discovery Methods
- Micro-level
- Dimensionality Reduction + Causal Discovery
- Subgroups
- Group Embedding

## Benchmarking
The benchmarking framework allows for easy comparison of different causal discovery methods. It includes:
- Generation of synthetic datasets based on time series directed acyclic graphs
- Test of algorithms in both static and dynamic hyperparameters conditions
- Plotting of results in atractive graphs
![image](https://github.com/user-attachments/assets/5fc44e5a-6488-4454-854d-e5737a290426)


## Contributing
Contributions are welcome! If you want to contribute:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License.

## Contact
For questions or collaborations, feel free to open an issue or contact us at [jmateosbarroso@gmail.com].


