from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

def read_VRP_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    print (f"lines = {lines}")
    num_customers = int(lines[0].split(":")[1].strip())
    vehicle_capacity = int(lines[1].split(":")[1].strip())
    depot_coordinates = tuple(map(int, lines[2].split(":")[1].strip().split(',')))
    
    # Skip the header line
    customer_data_start = lines.index("Customer coordinates and demand:\n") + 1

    # Read customer data
    customer_data = [list(map(int, line.strip().split(','))) for line in lines[customer_data_start:]]

    return num_customers, vehicle_capacity, depot_coordinates, customer_data

def create_graph(num_customers, depot_coordinates, customer_data):
    graph = nx.Graph()

    graph.add_node('depot', pos=depot_coordinates, demand=0)

    for i in range(num_customers):
        customer_id = f'customer_{i + 1}'
        customer_coordinates = tuple(customer_data[i][:2])
        demand = customer_data[i][2]
        graph.add_node(customer_id, pos=customer_coordinates, demand=demand)

    for i in range(num_customers + 1):
        for j in range(i + 1, num_customers + 1):
            node_i = 'depot' if i == 0 else f'customer_{i}'
            node_j = 'depot' if j == 0 else f'customer_{j}'
            distance = np.linalg.norm(np.array(graph.nodes[node_i]['pos']) - np.array(graph.nodes[node_j]['pos']))
            graph.add_edge(node_i, node_j, weight=distance)

    return graph    
    
class Graph:
    def __init__(self):
        """ default constructor: creates graphs adjancency matrix """
        self.adjacency_list = {}

    def add_edge(self, vertex1, vertex2, weight):
        """ function adds new adge into the graph  """
        edge = (vertex2, weight)
        if vertex1 in self.adjacency_list:
            self.adjacency_list[vertex1].append(edge)
        else:
            self.adjacency_list[vertex1] = [edge]

        if vertex2 not in self.adjacency_list:
            self.adjacency_list[vertex2] = []

    def display(self):
        """ function that displays adjencency matrix of graph  """
        for vertex, edges in self.adjacency_list.items():
            edge_str = ", ".join([f"{neighbor}({weight})" for neighbor, weight in edges])
            print(f"{vertex}: {edge_str}")

    def draw_graph(self):
        """ display graph """
        G = nx.DiGraph()
        for vertex, edges in self.adjacency_list.items():
            for neighbor, weight in edges:
                G.add_edge(vertex, neighbor, weight=weight)

        pos = nx.spring_layout(G)  # Layout to orginire nodes
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        labels = nx.get_edge_attributes(G, 'weight')

        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color="skyblue")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()
    