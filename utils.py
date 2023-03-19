import numpy as np
import qiskit
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeVigo

def IBM_load_account():
    qiskit.IBMQ.save_account("8bc4e3665e46ce69b7f8c4e9a0965749e9ea7b12a78803a6a0a1ae591b5ce2513a8f968c2a37391a0bc51ffc17b49c5935e375c66837b4d16a2ed8eda6c57ea0")
    qiskit.IBMQ.load_account()

def IBM_computer(circuit, provider_name, shots=1024):
    provider = qiskit.IBMQ.get_provider(hub='ibm-q-research',group='pernambuco-1', project='main')
    backend = provider.get_backend(provider_name)
    # transpiled_circs = transpile(circuit, backend=backend, optimization_level=3)
    transpiled_circs = transpile(circuit, backend=backend, optimization_level=3, initial_layout=[0, 1, 2])
    qobjs = assemble(transpiled_circs, backend=backend, shots=shots)
    job_info = backend.run(qobjs)

    results = []

    # get the results and append to the 'results' list
    for qcirc_result in transpiled_circs:
        results.append(job_info.result().get_counts(qcirc_result))

    return results

def get_res(cq, shots=1024):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    transpiled_circs = transpile(cq, backend=backend, optimization_level=3)
    qobjs = assemble(transpiled_circs, backend=backend,shots=shots)
    job_info = backend.run(qobjs)

    results = [job_info.result().get_counts(qcirc_result) for qcirc_result in transpiled_circs ]

    return results

def get_res_noisy(cq, shots=1024):

    backend = FakeVigo()
    sim_fake_vigo = AerSimulator.from_backend(backend)
    transpiled_circs = transpile(cq, backend=sim_fake_vigo, optimization_level=3)
    qobjs = assemble(transpiled_circs, backend=sim_fake_vigo,shots=shots)
    job_info = sim_fake_vigo.run(qobjs)

    results = [job_info.result().get_counts(qcirc_result) for qcirc_result in transpiled_circs ]

    return results

def accuracy(labels, predictions):

    loss = 0

    for l, p in zip(labels, predictions):
        if l == p:
            loss += 1
    loss = loss / len(labels)

    return loss

def inference(dic_measure):
    if not '1' in dic_measure:
        dic_measure['1'] = 0
    if not '0' in dic_measure:
        dic_measure['0'] = 0

    return (dic_measure['0'] - dic_measure['1'])/1024

def inference_array(dic_measure):
    inf = [inference(dic) for dic in dic_measure]
    return np.array(inf)

def ctrl_bin(state, level):

        state_bin = ''
        i = state
        while i//2 != 0:
            if(i>3):
                state_bin = str(i%2)+state_bin
                i = i//2
            else:
                state_bin = str(i//2)+str(i%2)+state_bin
                i = i//2

        #if level > len(state_bin):
        i = level - len(state_bin) - 1

        if state//2 == 0 and level > len(state_bin):
             state_bin = str(state%2)

        for _ in range(level-len(state_bin)):
            state_bin = '0'+state_bin

        return state_bin

def initializer(vetor):
    num_qu = int(np.log2(len(vetor))) + 1
    circuit = qiskit.QuantumCircuit(num_qu)

    norms = lambda v: np.sqrt(np.absolute(v[0::2])**2 + np.absolute(v[1::2])**2)
    select_alpha = lambda v,p,i: 2*np.arcsin(v[2*i + 1]/p[i]) if v[2*i]>0 else 2*np.pi - 2*np.arcsin(v[2*i + 1]/p[i])

    alphas = []
    parents = norms(vetor)
    alphas = np.append(alphas, np.array([ select_alpha(vetor,parents,i) for i in range(vetor.shape[0]//2)]))[::-1]

    for _ in range(int(np.log2(len(vetor)))-1):
        new_parents = norms(parents)
        alphas = np.append(alphas, np.array([ select_alpha(parents,new_parents,i) for i in range(parents.shape[0]//2)]))[::-1]
        parents = new_parents

    circuit.cry(alphas[0], [0], [1])
    circuit.cry((alphas[1]+alphas[2])/2, [0], [2])
    circuit.toffoli([0], [1], [2])
    circuit.cry((alphas[1] - alphas[2]) / 2, [0], [2])
    circuit.toffoli([0], [1], [2])

    return circuit

def initializer_model(vetor, params):
    num_qu = int(np.log2(len(vetor))) + 1
    circuit = qiskit.QuantumCircuit(num_qu)

    circuit.cry(params[0], [0], [1])
    circuit.cry((params[1] + params[2]) / 2, [0], [2])
    circuit.toffoli([0], [1], [2])
    circuit.cry((params[1] - params[2]) / 2, [0], [2])
    circuit.toffoli([0], [1], [2])

    return circuit

def split_data(X, Y, val, k_index):
    train_data = np.delete(X, range(k_index*val,(k_index+1)*val), axis=0)
    train_target = np.delete(Y, range(k_index*val,(k_index+1)*val))
    val_data = X[k_index*val:(k_index+1)*val,:]
    val_target = Y[k_index*val:(k_index+1)*val]

    return train_data, train_target, val_data, val_target

def shuffle_data(X,Y):
    shuf = np.array(range(Y.shape[0]))
    np.random.shuffle(shuf)

    return X[shuf], Y[shuf]
