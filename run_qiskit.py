import numpy as np
from qiskit_model import MQCOneClass
from utils import inference_array, accuracy, IBM_computer, IBM_load_account, get_res, get_res_noisy
from sklearn.model_selection import KFold
import csv

np.random.seed(13)

log_qiskit = []

def run_quantum(dataset_name, data, result_path, backend):
    # dataset_names = ['iris', 'wine', 'breast_cancer', 'balance', 'mammographic', 'banknote', 'voting']

    folds = ['fold_1.npy', 'fold_2.npy', 'fold_3.npy', 'fold_4.npy', 'fold_5.npy', 'fold_6.npy', 'fold_7.npy',
             'fold_8.npy', 'fold_9.npy', 'fold_10.npy']

    # best_fold = ['_fold_1.npy', '_fold_1.npy', '_fold_2.npy', '_fold_4.npy', '_fold_1.npy', '_fold_7.npy', '_fold_2.npy']

    # providers_names = ['ibmq_belem', 'ibm_perth', 'ibmq_jakarta']
    provider_name = 'ibmq_jakarta'

    indexes = np.array(range(len(data.y)))
    rkf = KFold(n_splits=10, shuffle=True, random_state=42)

    for f in range(len(folds)):
        print('FOLD: ', folds[f])
        # IBM_load_account()

        train_i, test_i = [[train, test] for train, test in rkf.split(indexes)][f]
        np.random.shuffle(train_i)
        np.random.shuffle(test_i)

        # params_path = 'params_final/' + dataset_name + '/' + folds[f]
        params_path = result_path + '/params/' + folds[f]
        var = np.load(params_path)

        inference_classes = np.zeros((var.shape[0], test_i.shape[0]))

        for i, vc in enumerate(var):
            class_array = []
            for sample in test_i:
                mqc_model = MQCOneClass(data.X[sample], vc, data.n_qfeatures)
                class_array.append(mqc_model.circuit)

            # dic_measure = {}

            if backend == 'noisy_simulation':
                dic_measure = get_res_noisy(class_array)
            # elif backend == 'noisy_free_simulation':
            #     dic_measure = get_res(class_array)
            # elif backend == 'real_quantum_computer':
            #     dic_measure = IBM_computer(class_array, provider_name)

            # dic_measure = get_res_noisy(class_array)
            inference_classes[i, :] = inference_array(dic_measure)

        inference_classes = inference_classes.transpose(1, 0)
        pred = np.array([np.argmax(p) for p in inference_classes])
        Y_val = [data.y[sample] for sample in test_i]
        acc_val = accuracy(Y_val, pred)
        print('Accuracy: ', acc_val)

        log_qiskit.append({'Model': backend,
                           'Dataset': dataset_name,
                           'Acc Val': acc_val})

    with open(result_path + '/qiskit_results.csv', 'w') as csv_file:
        fieldnames = ['Model', 'Dataset', 'Acc Val']
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        for values_dic in log_qiskit:
            writer_csv.writerow(values_dic)