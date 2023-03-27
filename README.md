# qVDCC
quantum Variational Distance-Based Centroid Classifier (qVDCC)

## Train the qVDCC
Perform the training procedure of qVDCC, searching for the best hyperparameters setting using [**Optuna**](https://optuna.org/) in each dataset.

    python3 train_pennylane.py

the output directories(of each dataset) contains the saved parameters, the cofusion matrix and csv(named `results.csv`) with accuracy results of each hyperparameters setting.

## Train Classic Algorithms
To perform the training procedure of KNN and Random Forest, searching for the best setting of hyperparameters using [**Optuna**](https://optuna.org/) in each dataset.

    python3 train_classic_classifiers.py

the output is a csv(named `classic_results.csv`) with accuracy results of each hyperparameters setting.

## Results Data Analysis and Cross Validation
After perform a data analysis to choose the best hyperparameters setting and perform a cross validation on the best models of each classification algorithm. `results_analysis.ipynb` can help with this analysis and generate the file:

- `complete_results.csv`: table with all hyperparameters setting with their respective accuracies. This file will be used in cross validation to choose the best model in cross validation procedure. To perform the cross validation of the best models run `python3 cross_validation.py`.
- `cross_validation.csv`: table with the results fo the cross validation of each algorithms models. This file if the output of `cross_validation.py`.

## Run on IBMQuantum Experience
Run the best trained model using [**Wine**](https://archive.ics.uci.edu/ml/datasets/Wine) dataset on a provider in IBMQ Experience to validate the qVDCC algorithm on real devices. This command run the inference and fine tuning the model in a real quantum device.

    python3 run_qiskit.py [provider_name]

To run this script, the configuration file `config.toml` of Pennylane framework has to be present on the `~/.config/pennylane` with your IBMQ and Qiskit settings. Useful links [Install Pennylane and Qiskit Pulugin for Pennylane](https://pennylane.ai/install.html), [Pennylane Plugins](https://pennylane.ai/plugins.html), [Docs for Qiskit Plugin for Pennylane](https://pennylaneqiskit.readthedocs.io/en/latest/), [Pennylane ibmq Device](https://pennylaneqiskit.readthedocs.io/en/latest/devices/ibmq.html), [Pennylane Configuration File](https://pennylane.readthedocs.io/en/latest/introduction/configuration.html).

## Python Dependencies

    python3 -m pip install -r requirements.txt
