from os.path import join
from typing import NamedTuple, Iterable, List
import sys

import numpy as np

class Edge(NamedTuple):
    head: int
    tail: int
    length: float

class Graph:
    def __init__(self, num_vertices: int, num_edges: int,
                 edges: Iterable[str],
                 starting_vertex_index: int=1):
        assert num_vertices > 0
        self._num_vertices = num_vertices
        self._num_edges = num_edges
        self.adj_mat = np.full((num_vertices, num_vertices), np.nan)
        for edge in edges:
            head, tail, length = edge.strip().split()
            head, tail, length = int(head), int(tail), float(length)
            self.adj_mat[head-starting_vertex_index,
                         tail-starting_vertex_index] = length

    def get_edge_length(self, head: int, tail: int) -> float:
        return self.adj_mat[head, tail]

    @property
    def adjacency_matrix(self):
        return self.adj_mat.copy()

    def get_all_edges(self) -> List[Edge]:
        raise NotImplementedError

    # TODO: add memoization here
    def get_incoming_edges(self, tail: int) -> Iterable[Edge]:
        tail_col = self.adj_mat[:, tail]
        with_edge = np.isfinite(tail_col)
        edge_costs = tail_col[with_edge]
        heads = np.argwhere(with_edge).flatten()
        edges = np.transpose(np.array([heads, np.fill(heads.shape, tail),
                                       edge_costs]))
        return [Edge(*values) for values in edges]

    def get_outgoing_edges(self, head: int) -> List[Edge]:
        raise NotImplementedError

    @property
    def num_vertices(self):
        return self._num_vertices

def generate_graph(file_name: str) -> Graph:
    with open(join('data', file_name), 'r') as f:
        num_vertices, num_edges = f.readline().strip().split()
        num_vertices, num_edges = int(num_vertices), int(num_edges)
        edges = f.readlines()
    return Graph(num_vertices, num_edges, edges)

def floyd_warshall(graph: Graph) -> np.ndarray:
    """
    Return a nxn matrix with shortest path for every (i, j) pair
    If there's a negative cost cycle, returns None
    """
    def has_negative_cycle(matrix: np.ndarray) -> bool:
        return (matrix.diagonal() < 0).any()
    num_vertices = graph.num_vertices
    result = graph.adjacency_matrix
    result[np.isnan(result)] = np.inf
    np.fill_diagonal(result, 0)
    for k in range(num_vertices):
        new = result.copy()
        for i in range(num_vertices):
            for j in range(num_vertices):
                new[i, j] = min([result[i, j], result[i, k] + result[k, j]])
        if has_negative_cycle(new):
            return None
        result = new
    return result

def get_shortest_shortest_path(graph: Graph) -> float:
    shortest_paths = floyd_warshall(graph)
    if shortest_paths is None:
        return None
    return shortest_paths.min()

def main(file_names: List[str]):
    result = []
    for file_name in file_names:
        graph = generate_graph(file_name.strip())
        ssp = get_shortest_shortest_path(graph)
        if ssp is not None:
            print("Shortest Shortest Path for {} is {}".format(file_name, ssp))
            result.append(ssp)
        else:
            print("Graph {} has at least 1 negative cycle".format(file_name))
    if result == []:
        print("All graphs have at least 1 negative cycle")
    else:
        print("Shortest Shortest Path among graphs is {}".format(min(result)))


if __name__ == '__main__':
    main(sys.argv[1:])
