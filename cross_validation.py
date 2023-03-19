from sklearn.model_selection import KFold
from circuits_pennylane import MQVClassifier
import csv
import sklearn
import numpy as np
import os
np.random.seed(13)
random_state = 42
# random_states = [42, 54, 78, 157, 420]

log_acc_data_mqvc = []
log_acc_data_knn = []

def accuracy(actual, pred):
    mask = np.array(actual==pred).astype(int)
    acc = np.sum(mask)/len(mask)

    return acc

def run_cross_validation(dataset_name, data, params_path, result_path):
    matrix_path = os.path.join(result_path, 'matrix')

    indexes = np.array(range(len(data.y)))
    rkf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1

    for train_i, test_i in rkf.split(indexes):
        print('Fold: ', fold)
        np.random.shuffle(train_i)
        np.random.shuffle(test_i)

        '''
        MQVC
        '''

        qexp = MQVClassifier(data.X[train_i],
                            data.y[train_i],
                            data.classes,
                            data.n_qfeatures,
                            test_data=data.X[test_i],
                            test_target=data.y[test_i])

        acc_train, acc_val = qexp.training( learning_rate=0.01,
                                            epochs=100,
                                            bs=1,
                                            threshold=1e-9,
                                            path=os.path.join(matrix_path, 'fold_' + str(fold)))

        np.save(os.path.join(params_path, 'fold_' + str(fold)), qexp.param.detach().numpy())

        log_acc_data_mqvc.append({'Model':'MQVC',
                                  'Dataset':dataset_name,
                                  'Acc Train':acc_train,
                                  'Acc Val':acc_val})

        '''
        KNN
        '''

        x_train, x_val, y_train, y_val = data.X[train_i], data.X[test_i], data.y[train_i], data.y[test_i]
        classifier_obj = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
        classifier_obj.fit(x_train, y_train)
        pred_train = classifier_obj.predict(x_train)
        pred_val = classifier_obj.predict(x_val)

        acc_train, acc_val = accuracy(y_train, pred_train), accuracy(y_val, pred_val)
        log_acc_data_knn.append({'Model': 'KNN',
                                 'Dataset': dataset_name,
                                 'Acc Train': acc_train,
                                 'Acc Val': acc_val})

        fold += 1

    with open(result_path + '/cross_validation.csv', 'w') as csv_file:
        fieldnames = ['Model', 'Dataset', 'Acc Train', 'Acc Val']
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        for values_dic in log_acc_data_mqvc:
            writer_csv.writerow(values_dic)
        for values_dic in log_acc_data_knn:
            writer_csv.writerow(values_dic)