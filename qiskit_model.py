import qiskit
from utils import initializer, initializer_model

class MQCOneClass():

    def __init__(self, x_test, params, n_qfeatures, measure=True):

        self._initialize(n_qfeatures)

        self.circuit.x(self.q[0])
        self._load_test(x_test)
        self.circuit.barrier()
        self.circuit.x(self.q[0])
        self.circuit.barrier()

        self._model_state(x_test, params)

        if measure:
            self.circuit.measure(self.q[0], self.c)

    def _initialize(self, n_qfeatures):
        self.n_qfeatures = n_qfeatures
        self.q = qiskit.QuantumRegister(self.n_qfeatures + 1)
        self.c = qiskit.ClassicalRegister(1)
        self.circuit = qiskit.QuantumCircuit(self.q, self.c)

        self.circuit.h(self.q[0])
        self.circuit.barrier()

    def _load_test(self, x_test):
        gate_preparation = initializer(x_test)
        self.circuit.compose(gate_preparation, inplace=True)
        self.circuit.barrier()

    def _model_state(self, x_test, params):
        gate_preparation_model = initializer_model(x_test, params)
        self.circuit.compose(gate_preparation_model, inplace=True)
        self.circuit.barrier()

        self.circuit.rz(params[3], [0])
        self.circuit.ry(params[4], [0])