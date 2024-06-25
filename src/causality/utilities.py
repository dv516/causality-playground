
from collections import defaultdict

import numpy as np
from scipy.stats import t
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
            


def adjacency_matrix_from_tstudent(t_student_values: pd.DataFrame, critical_value: float) -> pd.DataFrame:
    """
    Constructs an adjacency matrix based on t-student values using a critical value.

    Parameters:
    t_student_values (pd.DataFrame): DataFrame of t-student values.
    critical_value (float): Critical value to determine edges in the adjacency matrix.

    Returns:
    pd.DataFrame: Adjacency matrix where entries are 1/0 indicating edges between variables.
    """
    # Initialize an empty adjacency matrix with the same index and columns as t_student_values
    adjacency_matrix = pd.DataFrame(0, index=t_student_values.index, columns=t_student_values.columns)
    
    # Iterate through each cell in the DataFrame
    for i, row in t_student_values.iterrows():
        for j, value in row.iteritems():
            # Check if the absolute value of t-student value is greater than the critical value
            if abs(value) > critical_value:
                adjacency_matrix.loc[i, j] = 1
    
    return adjacency_matrix

def plot_correlation_graph(adjacency_matrix:pd.DataFrame,):

    G = nx.Graph()

    for col in adjacency_matrix.columns:
        G.add_node(col)

    for i in range(len(adjacency_matrix.columns)):
        for j in range(i + 1, len(adjacency_matrix.columns)):
            if adjacency_matrix.iloc[i, j]:
                G.add_edge(adjacency_matrix.columns[i], adjacency_matrix.columns[j], weight=adjacency_matrix.iloc[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # Edges
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.title('Correlation Graph')
    plt.show()

def plot_DAG_from_adjacency(adjacency_matrix):
    G = nx.DiGraph()

    for col in adjacency_matrix.columns:
        G.add_node(col)

    for i in range(len(adjacency_matrix.columns)):
        for j in range(len(adjacency_matrix.columns)):
            if adjacency_matrix.iloc[i, j]:
                G.add_edge(adjacency_matrix.columns[i], adjacency_matrix.columns[j], weight=adjacency_matrix.iloc[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # Edges
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.title('Causality DAG')
    plt.show()


# class Graph():
    
#     UNTOUCHED = 0
#     IN_PROGRESS = 1
#     DONE = 2

#     def __init__(self, adj_matrix):

#         self.n = len(adj_matrix)
#         self.children = defaultdict(list)
#         for i in range(self.n):
#             for j in range(self.n):
#                 if adj_matrix[i,j]:
#                     self.children[i].append(j)

#         self.status = [self.UNTOUCHED for _ in range(self.n)]
#         self.cycle = False

#     def is_DAG(self, ):
#         for starting_node in range(self.n):
#             if self.status[starting_node] == self.UNTOUCHED:
#                 if not self.cycle:
#                     self.BFS(starting_node)
#         return not self.cycle
    
#     def BFS(self, node):

#         self.status[node] = self.IN_PROGRESS

#         for neighbor in self.children[node]:
#             if self.status[neighbor] == self.IN_PROGRESS:
#                 self.cycle = True
#                 break
#             elif self.status[neighbor] == self.UNTOUCHED:
#                 self.BFS(neighbor)

#         self.status[node] = self.DONE

