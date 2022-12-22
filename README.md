# Neural network with random synaptic failures

Implementation of feed-forward neural networks with synaptic failures (analogous to *DropConnect* layers).

## Codes

- `networks`
	
	- `LinearWeightDropout` class: fully-connected layer, with random weight-drop (*DropConnect*); a random mask is generated for each input data point.

	- `Net` class: a base class for networks. It inherits from the `torch.nn.Module` class, with the addition of methods for the custom initialisation of the weights.

	- `LinearNet2L` and `LinearNet3L` classes: feed-forward neural networks, with 2 and 3 layers, respectively. By default, all layers are standar linear ones (`torch.nn.Linear`), but the `__init__` method allows to manually specify whether one or more layers should be replaced by another type (`layer_type` keyword argument).

	- `ClassifierNet2L` and `ClassifierNet3L` classes: analogous to the ones above (from which they inherit), but adds ReLU non-linearity for hidden layer and a softmax layer for the output one, specifically for classification tasks.

- `training_utils`:
	
	- `train_regressor` and `test_regressor` routines: train and test, respectively, a neural network model for a regression task

	- `train_classifier` and `test_classifier` routines: train and test, respectively, a neural network model for a classification task

- `data`:
	
	- `LinearRegressionDataset` class: a PyTorch `Dataset` defining output data to be linear in the input, with specified weights `w_star`; wrapping the data inside a `Dataset` class, allows one to use `DataLoader` to load batches of data. 

- `failures_LR` and `failures_MNIST`: main script for linear regression with linear output and for MNIST handwritten digit classification, respectively. The `plotting` section may not work properly, at the moment. See the `sys.argv` lines in the setup section, or `job_array_pars.txt` for the parameters to give as inputs.

- `stats_utils`: `run_statistics` function processing the weights, e.g. singular-value decomposition. 
