# Neural network with random synaptic failures

Implementation of feed-forward neural networks with synaptic failures (analogous to *DropConnect* layers).

## Codes

- `networks`.
	
	Objects for constructing custom neural network modules.
	
	- `LinearWeightDropout`:
	fully-connected layer, with random weight-drop (*DropConnect*); a random mask is generated for each input data point.

	- `Net`:
	a base class for networks. It inherits from the `torch.nn.Module` class, with the addition of methods for the custom initialisation of the weights.

	- `DeepNet`:
	deep feed-forward networks.
	It inherits from `Net`, and allows to stack different types of layers (fully-connected `torch.nn.Linear` by default), via keyword arguments 

	- `RNN`:
	vanilla recurrent neural network, with the option to replace recurrent layer with custom layer (e.g. `LinearWeightDropout`).

- `training_utils`:

	Routines for training and testing of neural networks.
	
	- `train_regressor`/`test_regressor`:
	train and test a neural network model for a regression task

	- `train_classifier`/`test_classifier`:
	train and test a neural network model for a classification task

- `data`
	
	Definitions of datasets, inheriting from  `torch.utils.data.Dataset`; this allows one to use `DataLoader` to load batches of data (see `training_utils`)
	
	- `LinearRegressionDataset`:
	linear target function, with specified weights `w_star`.

	- `SemanticsDataset`:
	target function used in

		> Saxe, A. M., McClelland, J. L. & Ganguli, S. A mathematical theory of semantic development in deep neural networks. *PNAS* **116**, 11537–11546 (2019).

		This is a linear target function specified as a function of a well-defined input-output covariance matrix, and of an input-input covariance matrix which can be passed as optional argument (identity by default).

- `stats_utils`: `run_statistics` function processing the weights, e.g. singular-value decomposition, and `load_statistics` to load the outputs.

Main scripts

- `failures_LR`:
	
	linear regression with linear target function.
	`pars_LR.py` generates the parameters to pass from command line.
	To run serially:
	```bash
	python pars_LR.py --run
	```
	To run in parallel (submit on SLURM system):
	```bash
	python pars_LR.py
	sbatch submit_LR.slurm
	```
