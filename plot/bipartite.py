import networkx as nx
import numpy as np
import pandas as pd


def draw(setA, setB, edges):
    G = nx.Graph()
    G.add_nodes_from(setA, bipartite=0)
    G.add_nodes_from(setB, bipartite=1)
    G.add_edges_from(edges)
    nx.draw_networkx(
        G,
        pos=nx.drawing.layout.bipartite_layout(G, setA),
        width=5)
