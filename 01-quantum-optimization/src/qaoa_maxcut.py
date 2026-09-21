"""Small MaxCut QAOA experiment.

The exact enumerator in maxcut.py remains the ground-truth reference.
This module uses Qiskit Optimization + QAOA for a small graph so that
approximation quality can be compared against the exact optimum.

This is an experiment scaffold, not a claim of quantum advantage.
"""

from __future__ import annotations

import networkx as nx
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut


def solve_qaoa(graph: nx.Graph, reps: int = 1, seed: int = 42):
    """Run QAOA on a small graph and return the optimization result."""
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("graph must contain at least one node")

    adjacency = nx.to_numpy_array(graph, nodelist=range(n), dtype=float)
    problem = Maxcut(adjacency).to_quadratic_program()

    sampler = StatevectorSampler(seed=seed)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=100),
        reps=reps,
    )
    optimizer = MinimumEigenOptimizer(qaoa)
    return optimizer.solve(problem)


if __name__ == "__main__":
    graph = nx.gnp_random_graph(6, 0.5, seed=42)
    result = solve_qaoa(graph, reps=1)
    print({
        "objective": float(result.fval),
        "solution": [int(x) for x in result.x],
        "reps": 1,
    })
