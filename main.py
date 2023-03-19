import os
import numpy as np
from dataset import LoadDataset
from cross_validation import run_cross_validation
from run_qiskit import run_quantum

np.random.seed(13)
dataset_names = ['iris', 'wine', 'breast_cancer', 'balance', 'mammographic', 'banknote', 'voting']
result_number = '01'
result_path = 'results/result_' + result_number

if __name__ == '__main__':

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        all_dir = [dir for dir in os.listdir('results/') if os.path.isdir(os.path.join('results/', dir))]
        max_result_number = max([int(number[-2:]) for number in all_dir])
        result_number = str(max_result_number + 1) if max_result_number >= 9 else '0' + str(max_result_number + 1)
        result_path = 'results/result_' + result_number
        os.makedirs(result_path)

    for dataset in dataset_names:
        print('DATASET:', dataset)
        data = LoadDataset(name=dataset)

        dataset_result_path = result_path + '/' + dataset
        dataset_image_path = os.path.join(dataset_result_path, 'images')
        dataset_charts_path = os.path.join(dataset_result_path, 'charts')
        dataset_params_path = os.path.join(dataset_result_path, 'params')

        os.makedirs(dataset_result_path)
        os.makedirs(dataset_image_path)
        os.makedirs(dataset_charts_path)
        os.makedirs(dataset_params_path)

        # Run cross-validation
        run_cross_validation(dataset, data, dataset_params_path, dataset_result_path)

        # Run Qiskit experiments
        run_quantum(dataset, data, dataset_result_path, 'noisy_simulation')