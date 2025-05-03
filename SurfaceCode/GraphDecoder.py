from operator import pos
# -*- coding: utf-8 -*-

"""
Graph decoder for surface codes
"""
import copy
import math
from itertools import combinations, product
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from networkx.exception import NetworkXNoPath





class GraphDecoder:
    """
    Class to construct the graph corresponding to the possible syndromes
    of a quantum error correction code, and then run suitable decoders.
    """

    def __init__(self, d, T, p ,simulation=False):
        self.d = d
        self.T = T
        self.p = p

        self.virtual = self._specify_virtual()
        self.S = {"X": nx.Graph(), "Z": nx.Graph()}
        self.D = nx.Graph()
        self.QX = nx.Graph()
        self.simulation = simulation
        self.left = -0.5
        self.top = -0.5
        self.right = 0
        self.bottom = 0
        self.final_bottom = 0
        self._make_syndrome_graph()
        self.left_most_nodes = []
        self.right_most_nodes = []

        for node in self.S["X"]:
            if node[2] == -0.5:
                self.left_most_nodes.append(node)
            else:
                self.right_most_nodes.append(node)

        self.final_right = self.right - 0.5
        self.final_bottom = self.bottom - 0.5
        self.data_qubit()

    def _specify_virtual(self):
        """Define coordinates of Z and X virtual nodes. Our convention is that Z
        virtual nodes are top/bottom and X virtual nodes are left/right.
        """
        virtual = {}
        virtual["X"] = []
        virtual["Z"] = []
        for j in range(0, self.d, 2):
            # Z virtual nodes
            virtual["Z"].append((-1, -0.5, j - 0.5))  # top
            virtual["Z"].append((-1, self.d - 0.5, j + 0.5))  # bottom

            # X virtual nodes
            virtual["X"].append((-1, j + 0.5, -0.5))  # left
            # self.left_most_virtual_nodes.append((-1, j + 0.5, -0.5))
            virtual["X"].append((-1, j - 0.5, self.d - 0.5))  # right
            # self.right_most_virtual_nodes.append((-1, j - 0.5, self.d - 0.5))
        return virtual

    def _make_syndrome_graph(self):
        start_nodes = {"Z": (0.5, 0.5), "X": (0.5, 1.5)}
        for error_key in ["X", "Z"]:
            # subgraphs for each time step
            for t in range(0, self.T):
                start_node = start_nodes[error_key]
                self.S[error_key].add_node(
                    (t,) + start_node,
                    virtual=0,
                    pos=(start_node[1], -start_node[0]),
                    time=t,
                    pos_3D=(
                        start_node[1],
                        -start_node[0],
                        t,
                    ),  # y-coord is flipped for plot purposes
                )
                self.populate_syndrome_graph(
                    (t,) + start_node, t, [], error_key, edge_weight=1
                )

            # connect physical qubits in same location across subgraphs of adjacent times
            syndrome_nodes_t0 = [
                x for x, y in self.S[error_key].nodes(data=True) if y["time"] == 0
            ]
            for node in syndrome_nodes_t0:
                space_label = (node[1], node[2])
                for t in range(0, self.T - 1):
                    self.S[error_key].add_edge(
                        (t,) + space_label, (t + 1,) + space_label, distance=1
                    )

    def populate_syndrome_graph(
        self, current_node, t, visited_nodes, error_key, edge_weight=1
    ):
        """Recursive function to populate syndrome subgraph at time t with error_key X/Z. The current_node
        is connected to neighboring nodes without revisiting a node.

        Args:
            current_node ((t, x, y)): Current syndrome node to be connected with neighboring nodes.
            visited_nodes ([(t, x, y),]): List of syndrome nodes which have already been traver.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.
            edge_weight (float, optional): Weight of edge between two adjacent syndrome nodes. Defaults to 1.

        Returns:
            None: function is to traverse the syndrome nodes and connect neighbors
        """
        visited_nodes.append(current_node)
        neighbors = []
        i = current_node[1]  # syndrome node x coordinate
        j = current_node[2]  # syndrome node y coordinate
        neighbors.append((i - 1, j - 1))  # up left
        neighbors.append((i + 1, j - 1))  # down left
        neighbors.append((i - 1, j + 1))  # up right
        neighbors.append((i + 1, j + 1))  # down right

        normal_neighbors = [
            n
            for n in neighbors
            if self.valid_syndrome(n, error_key)
            and (t, n[0], n[1]) not in visited_nodes
        ]  # syndrome node neighbors of current_node not already visited
        virtual_neighbors = [
            n
            for n in neighbors
            if (-1, n[0], n[1]) in self.virtual[error_key]
            and (-1, n[0], n[1]) not in visited_nodes
        ]  # virtual node neighbors of current_node not already visited

        # no neighbors to add edges
        if not normal_neighbors and not virtual_neighbors:
            return

        # add normal/non-virtual neighbors
        for target in normal_neighbors:
            target_node = (
                t,
            ) + target  # target_node has time t with x and y coordinates from target
            if not self.S[error_key].has_node(target_node):
                self.S[error_key].add_node(
                    target_node,
                    virtual=0,
                    pos=(target[1], -target[0]),
                    time=t,
                    pos_3D=(target[1], -target[0], t),
                )  # add target_node to syndrome subgraph if it doesn't already exist
                if(target[0] > self.bottom):
                  self.bottom = target[0]
                if(target[1] > self.right):
                  self.right = target[1]
            self.S[error_key].add_edge(
                current_node, target_node, distance=edge_weight
            )  # add edge between current_node and target_node

        # add virtual neighbors
        for target in virtual_neighbors:
            target_node = (
                -1,
            ) + target  # virtual target_node has time -1 with x and y coordinates from target
            if not self.S[error_key].has_node(target_node):
                self.S[error_key].add_node(
                    target_node,
                    virtual=1,
                    pos=(target[1], -target[0]),
                    time=-1,
                    pos_3D=(target[1], -target[0], (self.T - 1) / 2),
                )  # add virtual target_node to syndrome subgraph with z coordinate (T-1)/2 for nice plotting, if it doesn't already exist
            self.S[error_key].add_edge(
                current_node, target_node, distance=edge_weight
            )  # add edge between current_node and virtual target_node

        # recursively traverse normal neighbors
        for target in normal_neighbors:
            self.populate_syndrome_graph(
                (t,) + target, t, visited_nodes, error_key, edge_weight=1
            )

        # recursively traverse virtual neighbors
        for target in virtual_neighbors:
            self.populate_syndrome_graph(
                (-1,) + target, t, visited_nodes, error_key, edge_weight=1
            )

    def valid_syndrome(self, node, error_key):
        """Checks whether a node is a syndrome node under our error_key, which is either X or Z.

        Args:
            node ((t, x, y)): Node in graph.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.

        Returns:
            Boolean T/F: whether node is a syndrome node
        """
        i = node[0]
        j = node[1]
        if error_key == "Z":
            if i > 0 and i < self.d - 1 and j < self.d and j > -1:
                return True
            else:
                return False
        elif error_key == "X":
            if j > 0 and j < self.d - 1 and i < self.d and i > -1:
                return True
            else:
                return False

    def data_qubit(self):
      self.D = nx.Graph()
      self.left = 0
      self.bottom = 0
      self.top = self.d - 1
      self.right = self.d - 1

      # ➡️ Funzione per aggiungere nodi data_qubit
      def add_data_qubit_node(y, x, t):
          data_node = (-2, y, x)  # (-2, y, x)
          if not self.D.has_node(data_node):
              self.D.add_node(
                  data_node,
                  pos=(x, -y),
                  time=-2,
                  bit_flip = False,
                  phase_flip = False,
                  pos_3D=(x, -y, t),
                  type='data_q',
                  probability= self.p
              )
          return data_node

      # ➡️ Aggiungi nodi e archi partendo da S["X"]
      for node in self.S["X"]:
              self.D.add_node(
                  node,
                  pos=(node[2], -node[1]),
                  time=node[0],
                  pos_3D=(node[2], -node[1], node[0]),
                  type='X'
              )
              y, x = node[1], node[2]
              t = node[0]

              # Definisci i 4 data_qubit intorno
              bottom_left = (y + 0.5, x - 0.5)
              bottom_right = (y + 0.5, x + 0.5)
              top_left = (y - 0.5, x - 0.5)
              top_right = (y - 0.5, x + 0.5)

              # Aggiungi archi se dentro la griglia
              if (bottom_left[0] <= self.bottom + (self.d - 1)) and (bottom_left[1] >= self.left):
                  data_node = add_data_qubit_node(bottom_left[0], bottom_left[1], t)
                  self.D.add_edge(node, data_node)

              if (bottom_right[0] <= self.bottom + (self.d - 1)) and (bottom_right[1] <= self.right):
                  data_node = add_data_qubit_node(bottom_right[0], bottom_right[1], t)
                  self.D.add_edge(node, data_node)

              if (top_left[0] >= self.top - (self.d - 1)) and (top_left[1] >= self.left):
                  data_node = add_data_qubit_node(top_left[0], top_left[1], t)
                  self.D.add_edge(node, data_node)

              if (top_right[0] >= self.top - (self.d - 1)) and (top_right[1] <= self.right):
                  data_node = add_data_qubit_node(top_right[0], top_right[1], t)
                  self.D.add_edge(node, data_node)

      # ➡️ Aggiungi nodi e archi partendo da S["Z"]
      for node in self.S["Z"]:
              self.D.add_node(
                  node,
                  pos=(node[2], -node[1]),
                  time=node[0],
                  pos_3D=(node[2], -node[1], node[0]),
                  type='Z'
              )
              y, x = node[1], node[2]
              t = node[0]

              # Definisci i 4 data_qubit intorno
              bottom_left = (y + 0.5, x - 0.5)
              bottom_right = (y + 0.5, x + 0.5)
              top_left = (y - 0.5, x - 0.5)
              top_right = (y - 0.5, x + 0.5)

              # Aggiungi archi se dentro la griglia
              if (bottom_left[0] <= self.bottom + (self.d - 1)) and (bottom_left[1] >= self.left):
                  data_node = add_data_qubit_node(bottom_left[0], bottom_left[1], t)
                  self.D.add_edge(node, data_node)

              if (bottom_right[0] <= self.bottom + (self.d - 1)) and (bottom_right[1] <= self.right):
                  data_node = add_data_qubit_node(bottom_right[0], bottom_right[1], t)
                  self.D.add_edge(node, data_node)

              if (top_left[0] >= self.top - (self.d - 1)) and (top_left[1] >= self.left):
                  data_node = add_data_qubit_node(top_left[0], top_left[1], t)
                  self.D.add_edge(node, data_node)

              if (top_right[0] >= self.top - (self.d - 1)) and (top_right[1] <= self.right):
                  data_node = add_data_qubit_node(top_right[0], top_right[1], t)
                  self.D.add_edge(node, data_node)

    def inject_errors(self):
         for node in self.D.nodes:
            if self.D.nodes[node]["type"] == "data_q":
                if np.random.rand() < self.D.nodes[node]["probability"]:
                    self.D.nodes[node]["bit_flip"] = not self.D.nodes[node]["bit_flip"]
                if np.random.rand() < self.D.nodes[node]["probability"]:
                    self.D.nodes[node]["phase_flip"] = not self.D.nodes[node]["phase_flip"]

    def check_data_qubit_errors_X(self):
      list_x_error = []
      self.QX.add_nodes_from(self.S["X"])
      for node in self.D:
          parity = 0
          if self.D.nodes[node]["type"] == "X":
              for neighbor in self.D.neighbors(node):
                  if self.D.nodes[neighbor]["type"] == "data_q":
                      if self.D.nodes[neighbor]["bit_flip"]:
                          for x_syndromes in self.D.neighbors(neighbor):
                              if node != x_syndromes and self.D.nodes[x_syndromes]["type"] == "X":
                                if not self.QX.has_edge(*[node,x_syndromes]):
                                    self.QX.add_edge(*[node,x_syndromes])
                          parity += 1
              if parity % 2 == 1:
                  list_x_error.append(node)
      return list_x_error

    def check_data_qubit_errors_Z(self):
      list_z_error = []
      for node in self.D:
          parity = 0
          if self.D.nodes[node]["type"] == "Z":
              for neighbor in self.D.neighbors(node):
                  if self.D.nodes[neighbor]["type"] == "data_q":
                      if self.D.nodes[neighbor]["phase_flip"]:
                          parity += 1
              if parity % 2 == 1:
                  list_z_error.append(node)
      return list_z_error

    def make_error_graph(self, nodes, error_key, err_prob=None):
        """Creates error syndrome subgraph from list of syndrome nodes. The output of
        this function is a graph that's ready for minimum weight perfect matching (MWPM).

        If err_prob is specified, we adjust the shortest distance between syndrome
        nodes by the degeneracy of the error path.

        Args:
            nodes ([(t, x, y),]): List of changes of syndrome nodes in time.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.
            err_prob (float, optional): Probability of IID data qubit X/Z flip. Defaults to None.

        Returns:
            nx.Graph: Nodes are syndromes, edges are proxy for error probabilities
        """
        import networkx as nx

        paths = {}
        virtual_dict = nx.get_node_attributes(self.S[error_key], "virtual")
        time_dict = nx.get_node_attributes(self.S[error_key], "time")
        error_graph = nx.Graph()
        nodes += self.virtual[error_key]

        for node in nodes:
            if node not in self.S[error_key]:
                if node[0] == -1:
                    self.S[error_key].add_node(
                        node,
                        virtual=1,
                        pos=(node[2], -node[1]),
                        time=-1,
                        pos_3D=(node[2], -node[1], (self.T - 1) / 2),
                    )
            if not error_graph.has_node(node):
                error_graph.add_node(
                    node,
                    virtual=self.S[error_key].nodes[node]["virtual"],
                    pos=(node[2], -node[1]),
                    time=self.S[error_key].nodes[node]["time"],
                    pos_3D=(node[2], -node[1], self.S[error_key].nodes[node]["time"]),
                )

        for source, target in combinations(nodes, 2):
            try:
                distance = int(
                    nx.shortest_path_length(
                        self.S[error_key], source, target, weight="distance"
                    )
                )
            except nx.NetworkXNoPath:
                continue

            deg, path = self._path_degeneracy(source, target, error_key)
            if deg == 0:
                continue

            paths[(source, target)] = path

            if err_prob and err_prob > 0.0:
                denominator = math.log1p(-err_prob) - math.log(err_prob)
                if denominator == 0:
                    continue  # avoid division by zero
                distance = distance - math.log(deg) / denominator

            distance = -distance
            error_graph.add_edge(source, target, weight=distance)

        if self.simulation:
            return error_graph

        return error_graph, paths

    def analytic_paths(self, matches, error_key):
        analytic_decoder = GraphDecoder(self.d,self.T)
        paths = {}
        for (source,target) in matches:
            _, path = analytic_decoder._path_degeneracy(source[:3],target[:3], error_key)
            paths[(source[:3], target[:3])] = path
        return paths

    from networkx.exception import NetworkXNoPath

    def _path_degeneracy(self, a, b, error_key):
        """Calculate the number of shortest error paths that link two syndrome nodes
        through both space and time.

        Args:
            a (tuple): Starting or ending syndrome node (degeneracy is symmetric)
            b (tuple): Ending or starting syndrome node (degeneracy is symmetric)

        Returns:
            int: Number of degenerate shortest paths matching this syndrome pair
            [nodes,]: List of nodes for one of the shortest paths
        """
        # Select the correct syndrome subgraph
        if error_key == "X":
            subgraph = self.S["X"]
        elif error_key == "Z":
            subgraph = self.S["Z"]
        else:
            raise nx.exception.NodeNotFound("error_key must be X or Z")

        try:
            # Try to get all shortest paths
            shortest_paths = list(nx.all_shortest_paths(subgraph, a, b, weight="distance"))
        except NetworkXNoPath:
            # No path exists → degeneracy is zero
            return 0, []

        one_path = shortest_paths[0]  # Pick one representative path
        degeneracy = len(shortest_paths)

        # If either node is a virtual node, explore additional degeneracies to equivalent boundary virtuals
        source = None
        if a[0] == -1:
            target = a
            source = b
        elif b[0] == -1:
            target = b
            source = a

        # Include other virtual nodes at the same minimal distance
        if source:
            virtual_nodes = self.virtual[error_key]
            shortest_distance = nx.shortest_path_length(subgraph, source, target, weight="distance")
            for node in virtual_nodes:
                if node != target:
                    try:
                        distance = nx.shortest_path_length(subgraph, source, node, weight="distance")
                        if distance == shortest_distance:
                            additional_paths = list(nx.all_shortest_paths(subgraph, source, node, weight="distance"))
                            degeneracy += len(additional_paths)
                    except NetworkXNoPath:
                        continue  # Skip if not reachable

        return degeneracy, one_path


    def matching_graph(self, error_graph, error_key):
        """Return subgraph of error graph to be matched.

        Args:
            error_graph (nx.Graph): Complete error graph to be matched.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.

        Returns:
            nx.Graph: Subgraph of error graph to be matched
        """
        time_dict = nx.get_node_attributes(self.S[error_key], "time")
        subgraph = nx.Graph()
        syndrome_nodes = [
            x for x, y in error_graph.nodes(data=True) if y["virtual"] == 0
        ]
        virtual_nodes = [
            x for x, y in error_graph.nodes(data=True) if y["virtual"] == 1
        ]

        # add and connect each syndrome node to subgraph
        for node in syndrome_nodes:
            if not subgraph.has_node(node):
                subgraph.add_node(
                    node,
                    virtual=0,
                    pos=(node[2], -node[1]),
                    time=time_dict[node],
                    pos_3D=(node[2], -node[1], time_dict[node]),
                )
        for source, target in combinations(syndrome_nodes, 2):
            subgraph.add_edge(
                source, target, weight=error_graph[source][target]["weight"]
            )

        # connect each syndrome node to its closest virtual node in subgraph
        for source in syndrome_nodes:
            potential_virtual = {}
            for target in virtual_nodes:
                potential_virtual[target] = error_graph[source][target]["weight"]
            nearest_virtual = max(potential_virtual, key=potential_virtual.get)
            paired_virtual = (
                nearest_virtual + source
            )  # paired_virtual (virtual, syndrome) allows for the virtual node to be matched more than once
            subgraph.add_node(
                paired_virtual,
                virtual=1,
                pos=(nearest_virtual[2], -nearest_virtual[1]),
                time=-1,
                pos_3D=(nearest_virtual[2], -nearest_virtual[1], -1),
            )  # add paired_virtual to subgraph
            subgraph.add_edge(
                source, paired_virtual, weight=potential_virtual[nearest_virtual]
            )  # add (syndrome, paired_virtual) edge to subgraph

        paired_virtual_nodes = [
            x for x, y in subgraph.nodes(data=True) if y["virtual"] == 1
        ]

        # add 0 weight between paired virtual nodes
        for source, target in combinations(paired_virtual_nodes, 2):
            subgraph.add_edge(source, target, weight=0)

        return subgraph

    def matching(self, matching_graph, error_key):
        """Return matches of minimum weight perfect matching (MWPM) on matching_graph.

        Args:
            matching_graph (nx.Graph): Graph to run MWPM.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.

        Returns:
            [(node, node),]: List of matchings found from MWPM
        """
        matches = nx.max_weight_matching(matching_graph, maxcardinality=True)
        filtered_matches = [
            (source, target)
            for (source, target) in matches
            if not (len(source) > 3 and len(target) > 3)
        ]  # remove 0 weighted matched edges between virtual syndrome nodes
        return filtered_matches

    def calculate_qubit_flips(self, matches, paths, error_key):
        physical_qubit_flips = {}
        for (source, target) in matches:
            # Trim "paired virtual" nodes to nearest virtual node
            if len(source) > 3:
                source = source[:3]
            if len(target) > 3:
                target = target[:3]

            # Paths dict is encoded in one direction, check other if not found
            if (source, target) not in paths:
                source, target = (target, source)

            path = paths[(source, target)]  # This is an arbitrary shortest error path
            for i in range(0, len(path) - 1):
                start = path[i]
                end = path[i + 1]
                # Check if syndromes are in different physical locations
                # If they're in the same location, this is a measurement error
                if start[1:] != end[1:]:
                    time = start[0]
                    if time == -1:  # Grab time from non-virtual syndrome
                        time = end[0]
                    physical_qubit = (
                        -2,
                        (start[1] + end[1]) / 2,
                        (start[2] + end[2]) / 2,
                    )

                    
                    # Paired flips at the same time can be ignored
                    if physical_qubit in physical_qubit_flips:
                        physical_qubit_flips[physical_qubit] = (
                            physical_qubit_flips[physical_qubit] + 1
                        ) % 2
                    else:
                        physical_qubit_flips[physical_qubit] = 1
                    
                    if self.QX.has_edge(*[start,end]):
                        self.QX.remove_edge(*[start,end])
                    else:
                        self.QX.add_edge(*[start,end])

        physical_qubit_flips = {
            x: error_key for x, y in physical_qubit_flips.items() if y == 1
        }
        return physical_qubit_flips

    def net_qubit_flips(self, flips_x, flips_z):
        flipsx = {flip: "X" for flip, _ in flips_x.items() if flip not in flips_z}
        flipsz = {flip: "Z" for flip, _ in flips_z.items() if flip not in flips_x}
        flipsy = {flip: "Y" for flip, _ in flips_x.items() if flip in flips_z}
        flips = {**flipsx, **flipsy, **flipsz}

        individual_flips = defaultdict(dict)

        for flip, error_key in flips.items():
            individual_flips[flip[1:]][flip[0]] = error_key

        paulis = {
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
            "I": np.array([[1, 0], [0, 1]]),
        }

        physical_qubit_flips = {}
        for qubit_loc, flip_record in individual_flips.items():
            net_error = paulis["I"]
            # print("Physical Qubit: " + str(qubit_loc))
            for time, error in sorted(flip_record.items(), key=lambda item: item[0]):
                # print("Error: " + error + " at time: " + str(time))
                net_error = net_error.dot(paulis[error])
            physical_qubit_flips[qubit_loc] = net_error

        physical_qubit_flips = {x:y for x,y in physical_qubit_flips.items() if not np.array_equal(y,paulis["I"])}
        return physical_qubit_flips

    def apply_corrections(self, flips):
        for qubit in flips:
            self.D.nodes[qubit]["bit_flip"] = not self.D.nodes[qubit]["bit_flip"]

    def check_X_string(self, connected_component):
        """Returns True if connected_component represents a logical bit flip error."""
        left, right = False, False
        for node in connected_component:
            if not self.QX.degree[node] % 2 == 1:
                continue
            if node in self.left_most_nodes:
                left = True
            elif node in self.right_most_nodes:
                right = True
        return left and right
    
    def logical_operator_X(self):
        # result = 1
        # for node in self.D.nodes:
        #     if self.D.nodes[node]["type"] == "data_q":
        #         if node[2] == 0.0:  # x=0 side
        #             if(self.D.nodes[node]["bit_flip"]):
        #                 result *= -1
        #             else :
        #                 result *= 1 

        components = nx.connected_components(self.QX)
        logical_xs = 0
        for c in components:
            if len(c) > 1 and self.check_X_string(c):
                logical_xs += 1
        
        # if result == -1 and logical_xs%2==0:
        #     print(result)
        #     print(logical_xs)

        return int(logical_xs % 2 == 1) 

    def logical_operator_Z(self):
        result = True
        for node in self.D.nodes:
            if self.D.nodes[node]["type"] == "data_q":
                if node[1] == 0.0:  # check y = 0.0 line (bottom side typically for Z logical)
                    phase_flip = self.D.nodes[node]["phase_flip"]
                    result = result and phase_flip
        return result

    def graph_2D(self, G, edge_label):
      pos = nx.get_node_attributes(G, "pos")

      # 🎨 Draw nodes with borders and color
      nx.draw_networkx_nodes(G, pos, node_color='skyblue', edgecolors='black', node_size=600)

      # ➿ Draw edges
      nx.draw_networkx_edges(G, pos)

      # 🔢 Draw node labels (with index)
      node_labels = {
          node: f"{idx}: {node}" for idx, node in enumerate(G.nodes())
      }
      node_index = {node: idx for idx, node in enumerate(G.nodes())}
      nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

      # 🏷️ Draw edge labels if available
      try:
          edge_labels = nx.get_edge_attributes(G, edge_label)
          filtered_labels = {
              k: round(v, 2)
              for k, v in edge_labels.items()
              if isinstance(k[0], tuple) and isinstance(k[1], tuple)
          }
          nx.draw_networkx_edge_labels(G, pos, edge_labels=filtered_labels, font_size=7)
      except Exception as e:
          print(f"⚠️ Skipping edge labels due to error: {e}")

      plt.title(f"2D Graph View (Edges: {edge_label})")
      plt.axis('off')
      plt.show()

      return node_index

    def draw_nodes_by_type(self, G):
      pos = nx.get_node_attributes(G, "pos")

      # Prepara una lista di colori in base al tipo
      color_map = []
      for node, attr in G.nodes(data=True):
          node_type = attr.get("type", "unknown")
          if node_type == "data_q":
              color_map.append("green")
          elif node_type == "Z":
              color_map.append("red")
          elif node_type == "X":
              color_map.append("blue")
          else:
              color_map.append("gray")  # Se manca 'type', fallback

      # 🎨 Disegna nodi
      nx.draw_networkx_nodes(G, pos, node_color=color_map, edgecolors='black', node_size=600)

      # ➿ Disegna spigoli
      nx.draw_networkx_edges(G, pos)

      # 🏷️ Disegna etichette con posizione
      labels = {}
      for idx, (node, attr) in enumerate(G.nodes(data=True)):
          if isinstance(node, tuple):
              labels[node] = f"({round(node[1], 1)}, {round(node[2], 1)})"  # (x, y) formato
          else:
              labels[node] = str(idx)  # fallback

      nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

      plt.title("Graph with Positions as Labels")
      plt.axis('off')
      plt.show()

    def graph_3D(self, G, edge_label, angle=[-116, 22]):
        """Plots a graph with edge labels in 3D.

        Args:
            G (nx.Graph): Graph to plot in 3D.
            edge_label (float): Edge label to display; either distance or weight.
            angle ([float, float]): Initial 3D angle view. Defaults to [-116, 22]

        Returns:
            None: Plot is displayed in plt.show()
        """
        # Get node 3D positions
        pos_3D = nx.get_node_attributes(G, "pos_3D")

        # Define color range based on time
        colors = {
            x: plt.cm.plasma((y["time"] + 1) / self.T) for x, y in G.nodes(data=True)
        }

        # 3D network plot
        with plt.style.context(("ggplot")):

            fig = plt.figure(figsize=(20, 14))
            ax = Axes3D(fig)

            # Loop on the nodes and look up in pos dictionary to extract the x,y,z coordinates of each node
            for node in G.nodes():
                xi, yi, zi = pos_3D[node]

                # Scatter plot
                ax.scatter(
                    xi,
                    yi,
                    zi,
                    color=colors[node],
                    s=120 * (1 + G.degree(node)),
                    edgecolors="k",
                    alpha=0.7,
                )

                # Label node position
                ax.text(xi, yi, zi, node, fontsize=20)

            # Loop on the edges to get the x,y,z, coordinates of the connected nodes
            # Those two points are the extrema of the line to be plotted
            for src, tgt in G.edges():
                x_1, y_1, z_1 = pos_3D[src]
                x_2, y_2, z_2 = pos_3D[tgt]

                x_line = np.array((x_1, x_2))
                y_line = np.array((y_1, y_2))
                z_line = np.array((z_1, z_2))

                # Plot the connecting lines
                ax.plot(x_line, y_line, z_line, color="black", alpha=0.5)

                # Label edges at midpoints
                x_mid = (x_1 + x_2) / 2
                y_mid = (y_1 + y_2) / 2
                z_mid = (z_1 + z_2) / 2
                label = round(G[src][tgt][edge_label], 2)
                ax.text(x_mid, y_mid, z_mid, label, fontsize=14)

        # Set the initial view
        ax.view_init(angle[1], angle[0])

        # Hide the axes
        ax.set_axis_off()

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        plt.show()



