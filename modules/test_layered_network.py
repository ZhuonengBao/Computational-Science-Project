import unittest

from layered_network import LayeredNetworkGraph

Network = LayeredNetworkGraph([(1, 0, '')], 10, 1, inter_prob=0.0)


class TestLayeredNetwor(unittest.TestCase):
    def test_erdos_renyi_graph(self):
        n = 10
        p = 0.3
        G = Network.generate_erdos_renyi_digraph(n, p)
        self.assertEqual(len(G.nodes()), n,
                         "The number of created nodes does not equal N.")

        for u, v in list(G.edges()):
            self.assertFalse(G.has_edge(v, u),
                             "There are bidirectional connections between nodes")
    
    def test_one_layer(self):
        pass

    def test_two_layer(self):
        pass

    


if __name__ == '__main__':
    unittest.main(verbosity=2)
