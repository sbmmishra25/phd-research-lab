import itertools
import networkx as nx

def cut_value(graph, bits):
    return sum(bits[u] != bits[v] for u, v in graph.edges())

def exact_maxcut(graph):
    n = graph.number_of_nodes()
    best_bits, best_value = None, -1
    for bits in itertools.product((0, 1), repeat=n):
        value = cut_value(graph, bits)
        if value > best_value:
            best_bits, best_value = bits, value
    return best_bits, best_value

if __name__ == "__main__":
    g = nx.gnp_random_graph(8, .35, seed=42)
    print(exact_maxcut(g))
