# Created by Diego Alvarez-Estevez, 2025

# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Common utils

# External imports
from itertools import combinations
from sklearn import metrics
import pickle

# Qiskit dependences
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Print dataset number and class distributions
def count_labels(lab_vector):
    from collections import Counter
    a = dict(Counter(lab_vector))
    print(a)

# Creating covariant feature map from scratch
def CovariantFeatureMap(feature_dim, entanglement = None, reps=1):
    num_qubits = feature_dim / 2
    fm = QuantumCircuit(num_qubits)
    # If no entanglement scheme specified, use linear entanglement
    entangler_map = entanglement
    if entanglement is None:
        entangler_map = [[i, i + 1] for i in range(fm.num_qubits - 1)]
    if entanglement == "full":
        entangler_map = list(combinations(list(range(fm.num_qubits)), 2))
        #print(entangler_map)
    input_params = ParameterVector("x_par", feature_dim)
    # Create the entanglement layer
    for source, target in entangler_map:
        fm.cz(fm.qubits[source], fm.qubits[target])
    fm.barrier()
    # Create a circuit representation of the data group
    for i in range(fm.num_qubits):
        fm.rz(-2 * input_params[2 * i + 1], fm.qubits[i])
        fm.rx(-2 * input_params[2 * i], fm.qubits[i])
    # Process repetitions
    fm = fm.repeat(reps)
        
    return fm

def TrainableFeatureMap(adhoc_feature_map, training_params, qkt_single_training_parameter):
    # Using 3-param U-gate for general rotations on each qubit
    fm0 = QuantumCircuit(adhoc_feature_map.num_qubits)
    
    k = 0
    for i in range(adhoc_feature_map.num_qubits):
        if qkt_single_training_parameter == True:
            fm0.u(training_params[0], training_params[1], training_params[2], fm0.qubits[i])
        else:
            fm0.u(training_params[k], training_params[k+1], training_params[k+2], fm0.qubits[i])
            k += 3
    
    # Use (fixed) quantum mapping to represent input data
    fm1 = adhoc_feature_map # Start with one previously used
    
    # Create the feature map, composed of our two circuits
    fm = fm0.compose(fm1)
    
    # Print resulting circuit
    #print(circuit_drawer(fm))
    display(fm.draw("mpl"))
    
    print(f"QKT trainable parameters: {training_params}")
    
    return fm

class QKTCallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]
        self._it_num = 0

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)
        
        print(f"QKT callback, it: {self._it_num}, value: {x2}, params: {x1}")
        
        self._it_num += 1

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]

def compute_metrics(ytrain_true, ytrain_pred, ytest_true, ytest_pred):
     
    return {
        'accuracy_TR': metrics.accuracy_score(ytrain_true, ytrain_pred),
        'balanced_accuracy_TR': metrics.balanced_accuracy_score(ytrain_true, ytrain_pred),
        'cohen_kappa_TR': metrics.cohen_kappa_score(ytrain_true, ytrain_pred),
        'macro_f1_TR': metrics.f1_score(ytrain_true, ytrain_pred, average='macro'),
        'accuracy_TS': metrics.accuracy_score(ytest_true, ytest_pred),
        'balanced_accuracy_TS': metrics.balanced_accuracy_score(ytest_true, ytest_pred),
        'cohen_kappa_TS': metrics.cohen_kappa_score(ytest_true, ytest_pred),
        'macro_f1_TS': metrics.f1_score(ytest_true, ytest_pred, average='macro')
    }

def save_results(filename, objects):
    with open(filename, 'wb') as file:
        for obj in objects:
            pickle.dump(obj, file)

def load_results(filename):
    objects = []
    with open(filename, 'rb') as file:
        while True:
            try:
                objects.append(pickle.load(file))
            except EOFError:
                break
    return objects