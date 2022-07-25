from typing import Tuple, List
from collections import defaultdict

import numpy as np
from scipy import special

num_nodes = 20
epsilon = 1e-6

# Node 21 is taken to be the source direction for edges.
# Edges: (source, target), see for node IDs, and reduce index to 0-based
edges = [(j-1, i-1) for i, j in [
    (1, 2), (2, 3), (3, 3), (4, 3), (5, 3),
    (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), 
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)
]]


def node_adj_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    
    # Adjacency matrix
    directed_M = np.zeros((num_nodes, num_nodes), dtype='float32')
    undirected_M = np.zeros((num_nodes, num_nodes), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        directed_M[source_node,target_node] = 1
        undirected_M[source_node,target_node] = 1
        undirected_M[target_node,source_node] = 1
        
    # Normalization (directed)
    outgoing_degree = np.sum(directed_M,axis = 1)
    incoming_degree = np.sum(directed_M,axis = 0)
    directed_M_out = np.divide(directed_M,outgoing_degree,out = np.zeros_like(directed_M), where = outgoing_degree!=0) # Average importance to target nbrs
    directed_M_in = np.divide(directed_M,incoming_degree,out = np.zeros_like(directed_M), where = incoming_degree!=0) # Average importance to source nbrs
    directed_M = (directed_M_in + directed_M_out)/2 # Average importance of edge source-target
    directed_M = directed_M + epsilon
       
    # Normalization (undirected)
    all_degree = np.sum(undirected_M,axis = 0) # adjacency matrix is symmetric
    degree_matrix = np.zeros((num_nodes, num_nodes), dtype='float32')
    for i in range(num_nodes):
        degree_matrix[i,i] = all_degree[i]**(-0.5)
        
    undirected_M = np.matmul(degree_matrix,undirected_M)
    undirected_M = np.matmul(undirected_M,degree_matrix)
    undirected_M = undirected_M + epsilon

    return directed_M,undirected_M

class Graph():
    def __init__(self):
        self.num_nodes = num_nodes
        self.edges = edges
        # Adjacency matrices
        self.directed_M,self.undirected_M = node_adj_matrix(self.num_nodes, self.edges)