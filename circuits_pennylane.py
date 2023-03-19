from sklearn.model_selection import train_test_split
import numpy as np
import pennylane as qml
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import qiskit
from qiskit.providers.aer.noise import NoiseModel
import torch

def get_noise_model(provider_name):
    """"""""""""""""""""""""""""""""""""""""""""""""
    """" Load Qiskit settings to noisy circuits"""""
    """"""""""""""""""""""""""""""""""""""""""""""""

    qiskit.IBMQ.load_account()
    provider = qiskit.IBMQ.get_provider(group='open')
    backend = provider.get_backend(provider_name)
    coupling_map = backend.configuration().to_dict()["coupling_map"]
    noise_model = NoiseModel.from_backend(backend)
    return noise_model, coupling_map


class QuantumDevices():
    """ Store device settings

        Args:
            classes(list/numpy.array): Array with the class labels.
            n_qfeatures(int): Number of qubits to encode the input features.
            node_type(string): Type of pennylane node of device.
            provider_name(string): Provider name of IBMQ device.
    """
    def __init__(self,
                 classes,
                 n_qfeatures,
                 node_type='qubit',
                 provider_name = 'ibmq_qasm_simulator'):

        self.n_classes = len(classes)
        self.n_qfeatures = n_qfeatures
        self.n = self.n_classes * (self.n_qfeatures+1)
        self.node_dict = {'qubit':'default.qubit',
                          'mixed':'default.mixed',
                          'noisy':'qiskit.aer',
                          'ibmq':'qiskit.ibmq',
                          'aer':'qiskit.aer'}

        if node_type == 'ibmq':
            self.one_class_dev = qml.device('qiskit.ibmq',
                                            wires = self.n_qfeatures+1,
                                            backend=provider_name)
            # self.noise_model, self.coupling_map = get_noise_model(provider_name)

        elif node_type == 'noisy':
            self.noise_model, self.coupling_map = get_noise_model(provider_name)
            self.one_class_dev = qml.device(self.node_dict[node_type],
                                            wires = self.n_qfeatures+1,
                                            shots = 1000,
                                            noise_model = self.noise_model)
        else:
            self.one_class_dev = qml.device(self.node_dict[node_type],
                                            wires = self.n_qfeatures+1,
                                            shots = 1000)


    def transpile_set(self, opt_level=1, layout_method = "sabre", routing_method="sabre"):
        self.one_class_dev.set_transpile_args(
                                                    **{
                                                        "optimization_level": opt_level,
                                                        "coupling_map": self.coupling_map,
                                                        "layout_method": layout_method,
                                                        "routing_method": routing_method,
                                                    }
                                                )


class MQVariational(QuantumDevices):
    """ Defines the circuit of each classes module of MQVC.

        Args:
            classes(list/numpy.array): Array with the class labels.
            n_qfeatures(int): Number of qubits to encode the input features.
            node_type(string): Type of pennylane node of device.
            provider_name(string): Provider name of IBMQ device.
    """
    def __init__(self,
                 classes,
                 n_qfeatures,
                 node_type='qubit',
                 provider_name = 'ibmq_qasm_simulator'):

        super().__init__(classes,
                         n_qfeatures,
                         node_type=node_type,
                         provider_name=provider_name)

        self.readout_wires = 0 # readout qubit index
        self.input_wires = [1,2] # feature qubits index

        # Parameters to be learned
        self.param = np.random.rand(self.n_classes, 5)
        self.param = torch.tensor(self.param, requires_grad=True)

        self.one_class_circuit_train = qml.QNode(self.one_class_circuit,
                                                 self.one_class_dev,
                                                 interface="torch",
                                                 diff_method="parameter-shift")

    def one_class_circuit(self, input_feat, params, k):
        """Load the MQVC One-Class module circuit.

            Args:
                input_feat(numpy.array/Tensor): the array with input features.
                params: the parameters to load the model.
                k: the class of the module.
            Returns:
                qml.expval(Measurement)
        """

        self.one_class_input(input_feat) # Apply operators to encode the features
        self.one_class_model_circuit(params, k) # Apply operator with the model parameters

        return qml.expval(qml.PauliZ(0)) # Measurement

    def one_class_model_circuit(self, params, k):
        """Load the MQVC One-Class model circuit.
            Args:
                params: the parameters(sample features of the class k) to load the model.
                k: the class of the module.
        """

        qml.CRY(params[k,0], wires=[self.readout_wires, self.input_wires[0]])
        if self.n_qfeatures > 1:
            qml.CRY((params[k,1] + params[k,2]) / 2,
                    wires=[self.readout_wires, self.input_wires[1]])
            qml.Toffoli(wires=[self.readout_wires, self.input_wires[0], self.input_wires[1]])
            qml.CRY((params[k,1] - params[k,2]) / 2,
                    wires=[self.readout_wires, self.input_wires[1]])
            qml.Toffoli(wires=[self.readout_wires, self.input_wires[0], self.input_wires[1]])

        qml.RZ(params[k, 3], wires=self.readout_wires)
        qml.RY(params[k, 4], wires=self.readout_wires)


    def one_class_input(self, input_feat):
        """Load the MQVC One-Class input circuit.

            Args:
                input_feat(numpy.array/Tensor): the array with input features.
        """

        alphas = self.get_alphas(input_feat) # get the rotation angles
        self.stateprepare(alphas) # load feature states


    def stateprepare(self, alphas):
        """Load the MQVC One-Class feature states.

            Args:
                alphas(numpy.array): the array with rotation angles.
            Ref:
                arXiv:quant-ph/0406176
        """

        qml.Hadamard(wires=self.readout_wires)

        qml.PauliX(wires=self.readout_wires)
        qml.CRY(alphas[0],wires=[self.readout_wires, self.input_wires[0]])

        if self.n_qfeatures > 1:
            # Quantum 1 qubit multiplexer
            qml.CRY(( alphas[1] + alphas[2] ) / 2,
                    wires=[self.readout_wires,self.input_wires[1]])

            qml.Toffoli(wires=[self.readout_wires,
                               self.input_wires[0],
                               self.input_wires[1]])

            qml.CRY(( alphas[1] - alphas[2] ) / 2,
                    wires=[self.readout_wires,self.input_wires[1]])

            qml.Toffoli(wires=[self.readout_wires,
                               self.input_wires[0],
                               self.input_wires[1]])

        qml.PauliX(wires=self.readout_wires)


    def get_alphas(self, v_state):
        """Compute the MQVC One-Class rotation angles to load the input.

            Args:
                v_state(numpy.array/Tensor): the input array to be loaded.
            Returns:
                alphas(list/numpy): the array with th rotation angles.
            Ref:
                arXiv:quant-ph/0407010
        """
        n_qu = int(np.log2(len(v_state)))
        norms = lambda v: np.sqrt(np.absolute(v[0::2])**2 + np.absolute(v[1::2])**2)
        select_alpha = lambda v,p,i: 2*np.arcsin(v[2*i + 1]/p[i]) if v[2*i]>0 else 2*np.pi - 2*np.arcsin(v[2*i + 1]/p[i])

        alphas = []
        parents = norms(v_state)
        alphas = np.append(alphas, np.array([ select_alpha(v_state,parents,i) for i in range(v_state.shape[0]//2)]))[::-1]

        for _ in range(n_qu-1):
            new_parents = norms(parents)
            alphas = np.append(alphas, np.array([ select_alpha(parents,new_parents,i) for i in range(parents.shape[0]//2)])[::-1])
            parents = new_parents

        return alphas[::-1]


    def quantum_classifier(self, input_feat, params, k=0):

        return self.one_class_circuit_train(input_feat, params, k)


class MQVClassifier(MQVariational):
    """ Defines the class to train all MQVC One-Class modules.

        Args:
            X(numpy.array): the dataset to be used.
            y(numpy.array): the labels of the samples of the dataset.
            test_data(numpy.array): the feature set of test samples.
            test_data(numpy.array): the label set of test samples.
            classes(list/numpy.array): Array with the class labels.
            n_qfeatures(int): Number of qubits to encode the input features.
            node_type(string): Type of pennylane node of device.
            provider_name(string): Provider name of IBMQ device.
    """
    def __init__(self,
                 X,
                 y,
                 classes,
                 n_qfeatures,
                 test_data=None,
                 test_target=None,
                 test_size=0.1,
                 node_type = 'qubit',
                 provider_name = 'ibmq_qasm_simulator'):

        super().__init__(classes,
                         n_qfeatures,
                         node_type=node_type,
                         provider_name=provider_name)
        self.test_size = test_size

        self.labels = []
        for i in range(len(classes)):
            label = [-1 for _ in range(len(classes))]
            label[i] = 1
            self.labels.append(label)
        if test_data is None:
            self.load_data(X,y)
        else:
            self.X_train = X
            self.Y_train = y
            self.X_val = test_data
            self.Y_val = test_target

    def load_weights(self, params):
        """ Load the MQVC model weights.

            Args:
                params(list/numpy.array): the array with pre-treined parameters.
        """
        self.param = torch.tensor(params, requires_grad=True)

    def load_data(self, X, y):
        self.X = np.array(X)
        self.Y = y
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X,
                                                                              y,
                                                                              test_size=self.test_size,
                                                                              random_state=42)


    def cost(self, weights, input_feat, labels, k=0):
        """ Compute cost function of the One-Class module model
            to update the parameters values of rotations.

            Args:
                params(list/numpy.array): the array with pre-treined parameters.
            Returns:
                (Tensor): the mean squared loss cumputed.
        """

        predictions = [self.quantum_classifier(input_feat[f,:], weights, k).detach().numpy() for f in range(input_feat.shape[0])]

        return self.square_loss(labels, predictions)


    def square_loss(self, labels, predictions):
        """ Compute the mean squared loss.

            Args:
                labels(numpy.array/list/Tensor): the array with sample labels.
                predictions(numpy.arra/list/Tensor): the array with predictions of the model.
            Returns:
                (Tensor): the mean squared loss cumputed.
        """
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + np.sum(p-l) ** 2

        loss = loss / len(labels)

        return loss


    def complete_accuracy(self, labels, predictions):
        """ Compute the accuracy over MQVC One-Class modules predictions.

            Args:
                labels(numpy.array/list/Tensor): the array with sample labels.
                predictions(numpy.arra/list/Tensor): the array with predictions of the model.
            Returns:
                (float): the cumputed accuracy.
        """

        score = 0

        for l, p in zip(labels, predictions):
            p_class = np.argmax(p)
            l_class = np.argmax(l)
            if l_class == p_class:
                score += 1
        score = score / len(labels)

        return score


    def accuracy(self, labels, predictions):
        """ Compute the accuracy over one MQVC One-Class module predictions.

            Args:
                labels(numpy.array/list/Tensor): the array with sample labels.
                predictions(numpy.arra/list/Tensor): the array with predictions of the model.
            Returns:
                (float): the cumputed accuracy.
        """
        score = 0

        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                score = score + 1
        score = score / len(labels)

        return score


    def get_batch(self, Y_train,batch_size):
        """ Returns the train dataset and labels split in batches.

            Args:
                labels(numpy.array/list/Tensor): the array with sample labels.
                predictions(numpy.arra/list/Tensor): the array with predictions of the model.
            Returns:
                (float): the cumputed accuracy.
        """
        Y_train_batch = np.array([ Y_train[i:i+batch_size] for i in range(0,Y_train.shape[0], batch_size)])
        X_train_batch = np.array([ self.X_train[i:i+batch_size,:] for i in range(0,self.X_train.shape[0], batch_size)])

        return X_train_batch, Y_train_batch


    def get_predictions(self, k):
        """ Returns the MQVC One-Class module predictions.

            Args:
                k(int): the class of the module.
            Returns:
                predictions_train(list): the computed predictions(membership
                                         degree of the sample on the class k)
                                         of train set.
                predictions_val(list):  the computed predictions(membership
                                        degree of the sample on the class k)
                                        of validation set.
        """

        predictions_train = [self.quantum_classifier(self.X_train[f,:],self.param, k).detach().numpy() for f in range(self.X_train.shape[0])]
        predictions_val = [self.quantum_classifier(self.X_val[f,:],self.param, k).detach().numpy() for f in range(self.X_val.shape[0])]

        return predictions_train, predictions_val


    def training(self,
                 learning_rate = 0.01,
                 epochs = 50,
                 bs = 5,
                 threshold = 1e-5,
                 path = None):

        """ Perform the training procedure on each MQVC One-Class module and show
            the overall performance(cost/training accuracy/validation accuracy) of
            the MQVC.

            Args:
                learning_rate(float): the learning rate to be used in Adam optimizer.
                epochs(int): number of iterations over all the train dataset.
                bs(int): the batch size.
                threshold(float): the minimum error diference between previous
                                  and actual epoch cost.
                path(string): path to store the image of confusion matrix.
            Returns:
                acc_train(float): the computed accuracy of the complete model
                                  over the training set.
                acc_val(list):  the computed accuracy of the complete model
                                over the validation set.
        """

        # Instantiate the optimizer.
        opt = torch.optim.Adam([self.param], lr=learning_rate)
        batch_size = bs
        # Variable to store the best parameters setting of each module.
        best_var = self.param.detach().numpy()
        for k in range(self.n_classes):
            best_pred = 0
            old_cost = 0

            # Set the training labels for the current class module.
            print('TRAINING CLASS '+str(k))
            Y_train = np.array([1 if yi==k else -1 for yi in self.Y_train])
            Y_val = np.array([1 if yi==k else -1 for yi in self.Y_val])

            X_train_batch, Y_train_batch = self.get_batch(Y_train, batch_size)

            with tqdm(total=epochs) as t:
                for it in range(epochs):
                    for j in range(len(Y_train_batch)):
                        X_batch = torch.tensor(X_train_batch[j], requires_grad=False)
                        Y_batch = torch.tensor(Y_train_batch[j], requires_grad=False)

                        # Optimization step
                        def closure():
                            opt.zero_grad()

                            preds = torch.stack(
                               [self.quantum_classifier(f, self.param, k) for f in X_batch]
                            )

                            loss = torch.mean(torch.pow(preds-Y_batch, 2))
                            loss.backward()

                            return loss

                        opt.step(closure)

                    predictions_train, predictions_val = self.get_predictions(k)

                    acc_train = self.accuracy(Y_train, predictions_train)
                    acc_val = self.accuracy(Y_val, predictions_val)

                    if best_pred < acc_val:
                        best_pred = acc_val
                        best_var[k,:] = self.param[k,:].detach().numpy()

                    total_cost = self.cost(self.param, self.X_train, Y_train, k)
                    if abs(total_cost - old_cost) < threshold:
                        break
                    else:
                        old_cost = total_cost

                    postfix = {'Cost':total_cost,
                                'Acc_train':acc_train,
                                'Acc_val':acc_val}
                    t.set_postfix(postfix)
                    t.update()

        self.param = torch.tensor(best_var, requires_grad=True)

        acc_train, acc_val = self.serial_inference(path)

        return acc_train, acc_val


    def serial_inference(self, path=None):
        """ Perform(serially in each module) the inference on overall MQVC One-Class modules
            and show the overall performance(cost/training accuracy/validation accuracy) of
            the MQVC.

            Args:
                path(string): path to store the image of confusion matrix.
            Returns:
                acc_train(float): the computed accuracy of the complete model
                                  over the training set.
                acc_val(list):  the computed accuracy of the complete model
                                over the validation set.
        """

        Y_train = np.array([self.labels[yi] for yi in self.Y_train])
        Y_val = np.array([self.labels[yi] for yi in self.Y_val])

        inferences_train = []
        inferences_val = []

        for k in range(self.n_classes):
            t, v = self.get_predictions(k)
            inferences_train.append(t)
            inferences_val.append(v)

        predictions_train = np.array(inferences_train).transpose(1,0)
        predictions_val = np.array(inferences_val).transpose(1,0)

        y_model = np.array([np.argmax(p) for p in predictions_val])
        self.mat_val = confusion_matrix(self.Y_val, y_model)
        y_model = np.array([np.argmax(p) for p in predictions_train])
        self.mat_train = confusion_matrix(self.Y_train, y_model)

        if path != None:
            if not os.path.exists(path):
                os.makedirs(path)

            self.save_fig(self.mat_train, os.path.join(path,'train_mat.png'))
            self.save_fig(self.mat_val, os.path.join(path,'val_mat.png'))

        acc_train = self.complete_accuracy(Y_train, predictions_train)
        acc_val = self.complete_accuracy(Y_val, predictions_val)
        print(
            "Acc train: {:0.7f} | Acc validation: {:0.7f} "
            "".format( acc_train, acc_val)
        )

        return acc_train, acc_val

    def save_fig(self, mat, path):
        plt.figure()
        sns.heatmap(mat, square=True, annot=True, cbar=False)
        plt.xlabel('predicted value')
        plt.ylabel('true value');
        plt.savefig(path)