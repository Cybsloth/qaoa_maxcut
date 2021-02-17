"""QAOA for MaxCut problem"""
import math
from collections import Counter
import sys
from typing import Dict, Iterable, List, Tuple, Union

import cirq
import networkx as nx
import numpy as np
import optuna


optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.ERROR)


Node = Union[int, str]

class QAOAMaxCut:
    """QAOA for solving MaxCut problem."""
    def __init__(self,
                 depth: int = 4,
                 num_simulator_repetitions: int = 1000,
                 num_trials: int = 100,
                 random_state: int = 42):
        """
        Args:
            depth:
            num_simulator_repetitions:
            num_trials:
            random_state:

        """
        self._depth = depth
        self.random_state = random_state
        self.num_simulator_repetitions = num_simulator_repetitions
        self.num_trials = num_trials

        self._graph: nx.Graph
        self._qubits: List[cirq.GridQubit]
        self._circuit: cirq.Circuit

    @property
    def num_qubits(self) -> int:
        """Number of qubits (== numbers nodes )."""
        return len(self._graph.nodes)

    @property
    def graph(self) -> nx.Graph:
        """Get 'graph'."""
        return self._graph

    @graph.setter
    def graph(self, g: nx.Graph):
        """Set graph and create mappings 'graph's node' <-> 'qubit'."""
        self._graph = g
        n2q: Dict[Node, int] = {node: qubit for qubit, node in enumerate(g.nodes)}
        q2n: Dict[int, Node] = {qubit: node for node, qubit in n2q.items()}
        self._n2q = n2q
        self._q2n = q2n

    def __call__(self, graph: nx.Graph) -> Tuple[Dict[Node, int], float, float]:
        """Calculate QAOA for MaxCut problem on graph.

        Args:
            graph: Weighted graph.

        Returns:
            colored_vertexes: Sample split of vertexes (approx. solution).
            avg_cost: Average cost of states with optimum parameters.
            cost: Cost of most freq state.

        """
        self.graph = graph
        best_params = self._optimize_params()

        best_alphas = [v for k, v in best_params.items() if k.startswith('alpha')]
        best_gammas = [v for k, v in best_params.items() if k.startswith('gamma')]

        states = self._run_experiment(best_alphas, best_gammas)
        avg_cost = self._cost_function(states)

        mf_state = self._most_freq_state(states)
        mf_cost = self._state_cost(mf_state)

        colored_vertexes = {self._q2n[idx]: s for idx, s in enumerate(mf_state)}

        return colored_vertexes, -avg_cost, -mf_cost

    def _initialization_qubits(self):
        """Initialization of qubits."""
        self._qubits = [cirq.GridQubit(0, i) for i in range(self.num_qubits)]

        for qubit in self._qubits:
            yield cirq.H.on(qubit)

    def _cost_unitary(self, gamma: float):
        """Cost operator.

        Args:
            gamma: Parameter.

        """
        for edge in self.graph.edges:
            weight = self._edge_weight(edge)
            exp_degree = -gamma * weight / math.pi
            qubit_0, qubit_1 = map(lambda vertex: self._qubits[self._n2q[vertex]], edge)
            yield cirq.ZZPowGate(exponent=exp_degree).on(qubit_0, qubit_1)

    def _mixer_unitary(self, alpha: float):
        """Operator mi.

        Args:
            alpha: Parameter.

        """
        for qubit_idx in range(self.num_qubits):
            qubit = self._qubits[qubit_idx]
            yield cirq.XPowGate(exponent=-1*alpha/math.pi).on(qubit)

    def _create_circuit(self, alphas: List[float], gammas: List[float]):
        """Create quantum circuit.

        Args:
            alphas: Parameters for cost operator.
            gammas: Parameters for operator.

        """
        circuit = cirq.Circuit()
        circuit.append(self._initialization_qubits())

        for alpha, gamma in zip(alphas, gammas):
            circuit.append(self._cost_unitary(gamma))
            circuit.append(self._mixer_unitary(alpha))

        circuit.append(cirq.measure(*self._qubits, key='x'))

        self._circuit = circuit

    def _run_experiment(self, alphas: List[float], gammas: List[float]) -> np.ndarray:
        """Run experiment.

        Args:
            alphas:
            gammas:

        """
        self._create_circuit(alphas, gammas)
        simulator = cirq.Simulator(seed=self.random_state)
        results = simulator.run(self._circuit, repetitions=self.num_simulator_repetitions)
        result_measurements = results.measurements['x']

        return result_measurements

    def _cost_function(self, states: np.ndarray):
        """Calculate average cost of states.

        Args:
            states: States of qubits.

        Returns:
            cost: Average cost.

        """
        cost = 0.0
        for state in states:
            cost += self._state_cost(state)
        cost /= self.num_simulator_repetitions

        return cost

    def _state_cost(self, state: np.ndarray) -> float:
        """Calculate cost of state.

        Args:
            state: State of qubits.

        Returns:
            cost: Cost of state.

        """
        cost = 0.0
        for edge in self.graph.edges():
            weight = self._edge_weight(edge)
            s0, s1 = map(lambda vertex: 1 - 2*state[self._n2q[vertex]], edge)
            cost += 0.5 * weight * (s0 * s1  - 1)

        return cost

    def _optimize_params(self):
        """Optimization parameters of circuit."""
        def optuna_objective(trial: optuna.Trial):
            alphas, gammas = [], []
            for idx in range(self._depth):
                alpha = trial.suggest_uniform(f'alpha_{idx}', 0.0, 2*math.pi)
                gamma = trial.suggest_uniform(f'gamma_{idx}', 0.0, math.pi)

                alphas.append(alpha)
                gammas.append(gamma)

            states = self._run_experiment(alphas, gammas)
            cost = self._cost_function(states)

            return cost

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(optuna_objective, n_trials=self.num_trials)

        return study.best_params

    def _edge_weight(self, edge: Tuple[Node, Node]) -> float:
        """Get weight of edge."""
        return self.graph.get_edge_data(*edge)['weight']

    def _most_freq_state(self, states: Iterable[np.ndarray]) -> np.ndarray:
        """Get most frequency state.

        If there is more than one frequent state, choose first alphabetical order sort.

        Args:
            states: States of qubits.

        Returns:
            state: Most frequency state.

        """
        encode_states = {idx: state for idx, state in enumerate(states)}
        c = Counter(encode_states.keys())
        idx = c.most_common(1)[0][0]
        state = encode_states[idx]

        return state
