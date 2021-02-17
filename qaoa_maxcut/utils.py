"""Useful utilities."""
from itertools import product
import networkx as nx


def calculate_max_cut_cost(graph: nx.Graph) -> float:
    """Brute force MaxCut calculation.

    Args:
        graph: Weighted graph.

    Returns:
        max_cost: Maximum cost.

    """

    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes)}
    num_vertexes = len(node_to_idx)
    all_maxcut_iter = product(*((0,1) for _ in range(num_vertexes)))

    max_cost = 0.0

    for mask in all_maxcut_iter:
        cost = 0.0
        for edge in graph.edges:
            weight = graph.get_edge_data(*edge)['weight']
            cost += weight * (mask[node_to_idx[edge[0]]] != mask[node_to_idx[edge[1]]])
        if cost > max_cost:
            max_cost = cost

    return max_cost
