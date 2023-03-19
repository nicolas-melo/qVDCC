from sklearn import datasets, preprocessing, decomposition
import pandas as pd
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class LoadDataset():
    def __init__(self, name='iris'):

        if name == 'iris':
            self.classes = [0,1,2]
            self.features = [0,1,2,3]
            self.n_qfeatures = int(np.log2(len(self.features)))
            iris = datasets.load_iris()
            self.X = iris.data[:, self.features]
            self.y = iris.target

        elif name == 'wine':
            self.classes = [0,1,2]
            self.features = [0,1,2,3]
            self.n_qfeatures = int(np.log2(len(self.features)))
            wine = datasets.load_wine()
            pca = decomposition.PCA(n_components=4)
            self.X = pca.fit_transform(wine.data)
            self.y = wine.target

        elif name == 'breast_cancer':
            self.classes = [0,1]
            self.features = [0,1,2,3]
            self.n_qfeatures = int(np.log2(len(self.features)))
            breast_cancer = datasets.load_breast_cancer()
            pca = decomposition.PCA(n_components=4)
            self.X = pca.fit_transform(breast_cancer.data)
            self.y = breast_cancer.target
            self.X, self.y = self.resample(self.X, self.y)

        elif name == 'balance':
            self.classes = [0,1,2]
            self.features = [0, 1, 2, 3]
            self.n_qfeatures = int(np.log2(len(self.features)))
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'

            train_file_path = tf.keras.utils.get_file("balance.data.csv", TRAIN_DATA_URL)
            dataset = pd.read_csv(train_file_path).to_numpy()
            dataset[np.where(dataset[:,0]=='R')[0],0] = 0
            dataset[np.where(dataset[:,0]=='L')[0],0] = 1
            dataset[np.where(dataset[:,0]=='B')[0],0] = 2
            self.X, self.y = dataset[:,1:].astype(int), dataset[:,0].astype(int)

        elif name == 'mammographic':
            self.classes = [0,1]
            self.features = [0, 1, 2, 3]
            self.n_qfeatures = int(np.log2(len(self.features)))
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'

            train_file_path = tf.keras.utils.get_file("mammographic.data.csv", TRAIN_DATA_URL)
            dataset = pd.read_csv(train_file_path).dropna().to_numpy()
            pca = decomposition.PCA(n_components=4)
            data_fixes = []
            for row in dataset:
                if not '?' in row:
                    data_fixes.append(row.astype(int))
            data_fixes = np.array(data_fixes)
            dataset_data = pca.fit_transform(data_fixes[:,:-1])
            dataset_target = data_fixes[:,-1]
            self.X = dataset_data
            self.y = dataset_target

        elif name == 'voting':
            self.classes = [0,1]
            self.features = [0, 1, 2, 3]
            self.n_qfeatures = int(np.log2(len(self.features)))
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'

            train_file_path = tf.keras.utils.get_file("votes.data.csv", TRAIN_DATA_URL)
            dataset = pd.read_csv(train_file_path).dropna().to_numpy()
            pca = decomposition.PCA(n_components=4)
            dataset[np.where(dataset=='y')] = 1
            dataset[np.where(dataset=='n')] = -1
            dataset[np.where(dataset=='democrat')] = 1
            dataset[np.where(dataset=='republican')] = 0
            data_fixes = []
            for row in dataset:
                if not '?' in row:
                    data_fixes.append(row.astype(int))
            data_fixes = np.array(data_fixes)
            dataset_data = pca.fit_transform(data_fixes[:,1:])
            dataset_target = data_fixes[:,0]
            self.X = dataset_data
            self.y = dataset_target


        elif name == 'banknote':
            self.classes = [0,1]
            self.features = [0, 1, 2, 3]
            self.n_qfeatures = int(np.log2(len(self.features)))
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'

            train_file_path = tf.keras.utils.get_file("banknote.data.csv", TRAIN_DATA_URL)
            dataset = pd.read_csv(train_file_path).to_numpy()
            pca = decomposition.PCA(n_components=4)
            dataset_data = pca.fit_transform(dataset[:,:4])
            dataset_target = dataset[:,4]
            self.X, self.y = self.resample(dataset_data, dataset_target.astype(int))

        scaler = preprocessing.StandardScaler().fit(self.X)
        self.X = scaler.transform(self.X)
        self.X = preprocessing.normalize(self.X, norm='l2')

    def resample(self, X, Y, over_ratio=0.9, under_ratio=1.0):
        over = SMOTE(sampling_strategy=over_ratio)
        under = RandomUnderSampler(sampling_strategy=under_ratio)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_r, y_r = pipeline.fit_resample(X, Y)

        return X_r, y_r
