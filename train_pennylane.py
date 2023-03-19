import numpy as np
from circuits_pennylane import MQVClassifier
from dataset import LoadDataset
import os
import optuna
import csv
dataset_names = ['iris', 'wine', 'breast_cancer', 'balance', 'mammographic', 'banknote', 'voting']
log_acc_data = []

def objective(trial):

    qexp = MQVClassifier(data.X,
                        data.y,
                        data.classes,
                        data.n_qfeatures,
                        test_size=test_size)

    l = trial.suggest_float("lr", 0.01, 0.1, log=True)
    b = trial.suggest_int("batch_size", 1, 5, log=True)

    acc_train, acc_val = qexp.training( learning_rate=l,
                                        epochs=100,
                                        bs=b,
                                        threshold=1e-9,
                                        path=os.path.join(runs_matrix_path,'batch_'+str(b)+'lr_'+str(l)))

    print('acc_train: ', acc_train)
    print('acc_val: ', acc_val)

    np.save(os.path.join(runs_params_path, 'batch_' + str(b) + 'lr_' + str(l)), qexp.param.detach().numpy())
    log_acc_data.append({'Dataset':data_names,
                         'Learning Rate':l,
                         'Batch Size':b,
                         'Acc Train':acc_train,
                         'Acc Val':acc_val})

    return acc_val

for data_names in dataset_names:
    data = LoadDataset(name=data_names)
    data_names = 'results/'+data_names+'_torch'
    if not os.path.exists(data_names):
        os.makedirs(data_names)
    runs_matrix_path = os.path.join(data_names,'matrix')
    runs_params_path = os.path.join(data_names,'params')
    if not os.path.exists(runs_matrix_path):
        os.makedirs(runs_matrix_path)
    if not os.path.exists(runs_params_path):
        os.makedirs(runs_params_path)

    test_size = 0.3
    print("Dataset: ", data_names)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    print('study.best_trial: ', study.best_trial)

    with open(os.path.join(data_names,'results.csv'), 'w') as csv_file:
        fieldnames = ['Dataset', 'Learning Rate', 'Batch Size', 'Acc Train', 'Acc Val']
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        for values_dic in log_acc_data:
            writer_csv.writerow(values_dic)