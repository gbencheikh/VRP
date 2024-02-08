from utils import *
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# TODO : rho paramètre aléatoire 

class Ant:
    def __init__(self, graph, pheromone_matrix, alpha, beta, capacity):
        self.graph = graph
        self.pheromone_matrix = pheromone_matrix
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.tour = []


    def ant_tour(self, remaining_customers):
        self.tour = ['depot']
        current_capacity = 0

        while remaining_customers:
            current_node = self.tour[-1]
            probabilities = self.calculate_probabilities(current_node, remaining_customers)
            next_node = self.select_next_node(probabilities)
            customer_number = int(next_node.split('_')[1])
            demand = self.graph.nodes[next_node]['demand']

            if current_capacity + demand <= self.capacity:
                current_capacity += demand
                self.tour.append(next_node)
                remaining_customers.remove(customer_number)
            else:
                self.tour.append('depot')
                current_capacity = 0

        self.tour.append('depot')

    def calculate_probabilities(self, current_node, remaining_customers):
        index_customer = []
        pheromone_values = []
        heuristic_values = []

        for customer in remaining_customers:
            edge = (current_node, f'customer_{customer}')
            edge_data = self.graph.get_edge_data(*edge)

            if edge_data is not None:
                pheromone = self.pheromone_matrix.get(edge, 1.0)
                distance = edge_data['weight']
                heuristic = 1 / distance

                index_customer.append(customer)
                pheromone_values.append(pheromone)
                heuristic_values.append(heuristic)

        total_pheromones = sum(pheromone_values)
        total_heuristic = sum(heuristic_values)

        probabilities = [
            (index, (pheromone ** self.alpha) * (heuristic ** self.beta) / (total_pheromones ** self.alpha) / (total_heuristic ** self.beta))
            for i, (index, pheromone, heuristic) in enumerate(zip(index_customer, pheromone_values, heuristic_values))
        ]

        return probabilities

    def select_next_node(self, probabilities):
        if not probabilities:
            return 'depot'

        selected_index = random.choices(range(len(probabilities)), weights=[p[1] for p in probabilities])[0]
        selected_node = f'customer_{probabilities[selected_index][0]}'
        return selected_node

    def calculate_tour_cost(self):
        total_cost = 0
        for i in range(len(self.tour) - 1):
            node_i = self.tour[i]
            node_j = self.tour[i + 1]

            edge = (node_i, node_j)
            edge_data = self.graph.get_edge_data(*edge)
            
            distance = 0
            if edge_data is not None:
                distance = edge_data['weight']
            total_cost += distance

        return total_cost

class ACOVRP:
    def __init__(self, file_path, num_ants=10, evaporation_rate=0.5, alpha=1, beta=2):
        self.num_customers, self.vehicle_capacity, self.depot_coordinates, self.customer_data = read_VRP_input_file(file_path)
        print("Starting ACO...")

        print(f"Number of customers: {self.num_customers}")
        print(f"vehicle_capacity: {self.vehicle_capacity}")
        print(f"Depot_coordinates: {self.depot_coordinates}")
        print(f"Customer_data: {self.customer_data}")

        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.graph = create_graph(self.num_customers, self.depot_coordinates, self.customer_data)
        self.pheromone_matrix = self.initialize_pheromones()

    def ant_colony(self):
        ants = [Ant(self.graph, self.pheromone_matrix, self.alpha, self.beta, self.vehicle_capacity) for _ in range(self.num_ants)]
        remaining_customers = set(range(1, self.num_customers + 1))

        for ant in ants:
            ant.ant_tour(remaining_customers)

        return ants

    def initialize_pheromones(self):
        pheromones = {(i, j): 1.0 for i in range(self.num_customers + 1) for j in range(self.num_customers + 1) if i != j}

        for i in range(1, self.num_customers + 1):
            pheromones[('depot', f'customer_{i}')] = 1.0
            pheromones[(f'customer_{i}', 'depot')] = 1.0

        for i in range(1, self.num_customers + 1):
            for j in range(1, self.num_customers + 1):
                if i != j:
                    pheromones[(f'customer_{i}', f'customer_{j}')] = 1.0
                    pheromones[(f'customer_{j}', f'customer_{i}')] = 1.0

        return pheromones

    def calculate_delta_pheromones(self, tour, cost):
        delta_pheromones = {}
        for i in range(len(tour) - 1):
            edge = (tour[i], tour[i + 1])
            delta_pheromones[edge] = 1.0 / cost
        return delta_pheromones

    def update_pheromones(self, delta_pheromones, evaporation_rate):
        for edge, delta in delta_pheromones.items():
            self.pheromone_matrix[edge] = (1 - evaporation_rate) * self.pheromone_matrix[edge] + delta

    def solve(self, num_iterations=100):
        best_solution = None
        best_cost = float('inf')
        self.pheromone_matrix = self.initialize_pheromones()

        for iteration in range(num_iterations):
            ants = self.ant_colony()

            for ant in ants:
                cost = ant.calculate_tour_cost()

                if cost != 0:
                    if cost < best_cost:
                        best_solution = ant.tour
                        best_cost = cost

                    delta_pheromones = self.calculate_delta_pheromones(ant.tour, cost)
                    self.update_pheromones(delta_pheromones, self.evaporation_rate)

            print(f"Iteration {iteration + 1}, Best Cost: {best_cost}")

        return best_solution, best_cost

def plot_graph_with_solution(graph, solution):
    pos = nx.get_node_attributes(graph, 'pos')

    # Plot nodes without labels and edges
    nx.draw(graph, pos, with_labels=False, node_size=150, node_color='lightblue', edgecolors='white', width =0.1)

    colors = ['red', 'orange', 'green', 'blue', 'purple']

    tour = []
    index = 0
    for id, node in enumerate(solution):
        tour.append(node)
        if id != 0 and node == 'depot':
            # Highlight the solution edges
            solution_edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
            nx.draw(graph, pos, edgelist=solution_edges, edge_color=colors[index], width=2, with_labels=False, node_size=150, node_color='lightblue', edgecolors='black', linewidths=0.8)
            index += 1
            tour = [node]

    # Add labels for customers without overlap
    labels = {node: node.split('_')[1] if 'customer' in node else node for node in graph.nodes()}
    pos_lower = {k: (x, y - 0.05) for k, (x, y) in pos.items()}
    nx.draw_networkx_labels(graph, pos_lower, labels, font_size=8, font_color='black', verticalalignment="bottom")

    # Add labels for demands
    demand_labels = {node: f'({graph.nodes[node]["demand"]})' for node in graph.nodes() if node != 'depot'}
    pos_higher = {k: (x, y - 0.15) for k, (x, y) in pos.items()}
    nx.draw_networkx_labels(graph, pos_higher, demand_labels, font_size=6, font_color='blue', verticalalignment="bottom")

    plt.title('Vehicle Routing Problem - Solution')
    plt.show()

if __name__ == "__main__":
    filepath = "VRP\\ACO\\data.txt"
    aco_instance = ACOVRP(filepath)
    best_solution, best_cost = aco_instance.solve()
    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)

    # Plot the graph with the best solution
    plot_graph_with_solution(aco_instance.graph, best_solution)