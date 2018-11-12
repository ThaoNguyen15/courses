from unittest import TestCase

import numpy as np

import shortest_shortest_path as ssp

class TestFloydWarshall(TestCase):

    def test_single_vertice(self):
        pass

    def test_negative_cycle(self):
        g = ssp.generate_graph('neg_cycle.txt')
        shortest_paths = ssp.floyd_warshall(g)
        self.assertIsNone(shortest_paths)

    def test_cycle(self):
        pass

    def test_simple(self):
        g = ssp.generate_graph('simple.txt')
        shortest_paths = ssp.floyd_warshall(g)
        correct_ssp = np.array([[0, -1, -3, -3],
                                [np.inf, 0, -2, -2],
                                [np.inf, 2, 0, 0],
                                [np.inf, 2, 0, 0]])
        self.assertTrue(np.array_equal(shortest_paths, correct_ssp))


class TestGraph(TestCase):

    def test_simple(self):
        g = ssp.generate_graph('simple.txt')
        adj_mat = np.array([[np.nan, -1., 1., np.nan],
                            [np.nan, np.nan, np.nan, -2.],
                            [np.nan, 2., np.nan, np.nan],
                            [np.nan, np.nan, 0., np.nan]])
        np.testing.assert_equal(g.adjacency_matrix, adj_mat)
        self.assertEqual(g.num_vertices, 4)

    def test_no_vertice(self):
        with self.assertRaises(AssertionError):
            ssp.Graph(0, 0, [])
