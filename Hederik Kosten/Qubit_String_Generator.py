import pennylane as qml
import numpy as np

def make_circuit(num_qubits, decay=1):
    dev = qml.device("default.qubit", wires=num_qubits)

    def circuit_fn():
        for i in range(num_qubits):
            qml.RX(np.pi / (i+1) ** decay, wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        #TEMP
        for i in range(1, num_qubits):
            qml.CNOT(wires=[i, i-1])

    @qml.qnode(dev, shots=1)
    def sampling_circuit():
        circuit_fn()
        return qml.sample()

    return sampling_circuit

def generate_qubit_samples(num_samples, num_qubits, decay=1):
    circuit = make_circuit(num_qubits, decay)
    result = np.zeros((num_samples, num_qubits), dtype=int)
    for i in range(num_samples):
        result[i] = circuit()[0]
    return result

# print(generate_qubit_samples(1, 10))