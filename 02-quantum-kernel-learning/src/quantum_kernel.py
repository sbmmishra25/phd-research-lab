"""Research-grade quantum-kernel baseline.

Uses the current Qiskit Machine Learning primitive-based API:
FidelityQuantumKernel + QSVC.
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

def run(seed=42):
    X, y = make_moons(n_samples=80, noise=0.15, random_state=seed)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    feature_map = zz_feature_map(feature_dimension=2, reps=2, entanglement="linear")
    sampler = StatevectorSampler(seed=seed)
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    model = QSVC(quantum_kernel=kernel)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "qubits": feature_map.num_qubits,
        "circuit_depth": feature_map.depth(),
        "trainable_parameters": feature_map.num_parameters,
    }

if __name__ == "__main__":
    print(run())
