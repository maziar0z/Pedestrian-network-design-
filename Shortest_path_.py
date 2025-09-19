#! python 3

"""Chapter 2 -- Shortest paths dijkstra and A* only"""

from math import inf
import logging
from copy import deepcopy
from collections import OrderedDict
from time import time

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


import heapq

cons = 10  # length threshold



# update the graph by checking y for links
def revise_graph_by_y(graph, y):
    """
    Removes unavailable links from a graph based on y[link] = 0.

    Parameters:
        graph : dict
            Original adjacency list {node: [(neighbor, weight), ...], ...}
        y : dict
            Dictionary with link availability, e.g., {'AB': 1, 'AC': 0, ...}

    Returns:
        new_graph : dict
            Revised graph with only active links
    """
    new_graph = {}

    for u in graph:
        new_graph[u] = []
        for v, weight in graph[u]:
            link_name = f"{u}{v}"
            if y.get(link_name, 1) == 1:  # keep if y[link]==1 or not specified
                new_graph[u].append((v, weight))
    
    return new_graph






def dijkstra(graph, source, y):
    """
    Finds shortest paths from source to all nodes using Dijkstra's algorithm,
    considering only links with y[link] = 1.

    Parameters:
        graph (dict): Adjacency list representation of the graph.
                      Example: {u: [(v1, w1), (v2, w2), ...], ...}
        source (any): Source node
        y (dict): Dictionary of link availability. 
                  Keys are strings like 'AB', values 0/1.

    Returns:
        dist (dict): Shortest distance from source to each node.
        prev (dict): Previous node in the shortest path.
    """
    dist = {node: float('inf') for node in graph}
    dist[source] = 0
    prev = {node: None for node in graph}

    # Priority queue: (distance, node)
    queue = [(0, source)]

    while queue:
        current_dist, u = heapq.heappop(queue)

        # Skip if this entry is outdated
        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:
            link = u + v  # construct link name like 'AB'
            # print( " we are looking at origin " , source, " current node is " , v )
            # if farther than threshold, just return 100000
            if current_dist + weight > cons:
                dist[v] = 100000
                prev[v] = u
                continue

            # Only traverse if link is available
            if y.get(link, 0) == 1:
                alt = current_dist + weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(queue, (alt, v))

    return dist, prev



def find_shortest_route(graph, start, end, y):
    """
    Finds the shortest route from start to end in the graph.
    
    Parameters:
        graph (dict): Adjacency list representation of the graph.
        start (any): Starting node
        end (any): Ending node
    
    Returns:
        path (list): List of nodes in the shortest path from start to end.
    """
    # print(" solution of dijkstra is ", dijkstra(graph, start), " start is ", start, " end is ", end )
    dist, prev = dijkstra(graph, start, y)
    length = dist[end]
    path = []
    current = end
    while current is not start:
        path.append(current)
        if prev[current] is None:
            return [], inf
        else:
            current = prev[current]
    
    path.append(start)
    path.reverse()  # Reverse to get the path from start to end
    return path, length 
    


# find the link ids that are in the shortest path
def find_links_in_shortest_path(graph, start, end, y):
    """
    Finds the links in the shortest path from start to end in the graph.
    
    Parameters:
        graph (dict): Adjacency list representation of the graph.
        start (any): Starting node
        end (any): Ending node
    
    Returns:
        links (list): List of links in the shortest path from start to end.
    """
    path, _ = find_shortest_route(graph, start, end, y)
    links = []
    for i in range(len(path) - 1):
        links.append(f"{path[i]}{path[i+1]}")
    return links



# def main():

#     # Acyclic Network
#     A = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]
#     L = {(1, 2): 2, (1, 3): 4, (2, 3): 1, (2, 4): 5, (3, 4): 2}
#     # heuristic = {1: 3, 2: 2, 3: 1, 4: 0}
    
#     # As = [['1', '2'], ['1', '3'], ['2', '3'], ['2', '4'], ['3', '4']]
#     Ls = {('1', '2'): 2, ('1', '3'): 4, ('2', '3'): 1, ('2', '4'): 5, ('3', '4'): 2}
#     heuristic_s = {'1': 3, '2': 2, '3': 1, '4': 0}
    
#     SP = Shortest_path(start='1', labels=Ls, heuristic=heuristic_s)

#     for algorithm in [SP.dijkstra, SP.a_star]:
#         if algorithm== SP.dijkstra:
#             print("dijkastra:")
#             L, q = algorithm()
#             print(f'L_r: {L}\nq_r: {q}\n')
            
#         else:
#             print("a_star:")
#             L, q = algorithm()
#             print(f'L_r: {L}\nq_r: {q}\n')

#     return SP


def find_all_paths(graph, source, target, path=None):
    if path is None:
        path = []

    path = path + [source]

    if source == target:
        return [path]

    if source not in graph:
        return []

    paths = []
    for neighbor, _ in graph[source]:
        if neighbor not in path:  # avoid cycles
            new_paths = find_all_paths(graph, neighbor, target, path)
            for p in new_paths:
                paths.append(p)
    return paths



def find_all_nodes(graph):
    """  Returns a list of nodes in the graph. """
    nodes = set()
    for u in graph:
        nodes.add(u)
    return list(nodes)


def find_all_links(graph):
    """
    Given a graph as an adjacency list {u: [(v, weight), ...], ...},
    return a list of all directed links as strings "uv".
    """
    links = []
    for u in graph:
        for v, _ in graph[u]:
            links.append(f"{u}{v}")
    return links


# # find cost for a given od using dijakstra
# def find_cost_od(graph, nodes, origin, y):
# # finds the cost from origin to all other nodes
#     dist, _ = dijkstra(graph, origin, y)
    
#     # Filter for only nodes in the graph to be safe
#     cost_dict = {node: dist.get(node, float('inf')) for node in nodes if node != origin}
#     return cost_dict



    

# def find_od_links(graph, origin, destination):
#     """
#     Given a graph as an adjacency list {u: [(v, weight), ...], ...},
#     and an origin and destination node, return the list of all unique
#     links (as strings 'uv') that appear on any path from origin to destination.
#     """
#     # First, get all simple paths from origin to destination:
#     paths = find_all_paths(graph, origin, destination)

#     # Now extract every link along those paths
#     link_set = set()
#     for path in paths:
#         for i in range(len(path) - 1):
#             u, v = path[i], path[i+1]
#             link_set.add(f"{u}{v}")

#     # Return as a sorted list (optional)
#     return sorted(link_set)



def find_edges_cost(graph):
    """ Returns a list of edges in the graph. 
    format: [(begin, end, cost), ...]
    """
    edges = set()
    for u in graph:
        for v, weight in graph[u]:
            edges.add((u, v, weight))
    return list(edges)








# for a given rs (origin destination), find the h_pi list
def find_h_pi(graph, origin, destination, y):
    """
    Finds the h_pi list for a given origin and destination in the graph.
    
    Parameters:
        graph (dict): Adjacency list representation of the graph.
        origin (any): Origin node
        destination (any): Destination node
    
    Returns:
        h_pi (list): List of heuristic values from origin to destination.
    """
    # dist, prev = dijkstra(graph, origin)
    paths = find_all_paths(graph, origin, destination)
    short_path = find_shortest_route(graph, origin, destination, y)[0]
    
    h_pi = {}
    current = destination
    for path in paths:
        path_tuple = tuple(path)
        if path == short_path:
            h_pi[path_tuple] = 1
        else:
            h_pi[path_tuple] = 0
    
    return h_pi


# def find_f_od(graph, origin, destination, threshold, y):
#     """
#     Finds the f_od binary value for a given origin and destination in the graph.
    
#     Parameters:
#         graph (dict): Adjacency list representation of the graph.
#         origin (any): Origin node
#         destination (any): Destination node
#         c (int): length threshold for the path
    
#     Returns:
#         f_od (binary): whether the destination is reachable within the length threshold c.
#     """
#     length = find_shortest_route(graph, origin, destination, y)[1]
#     if length <= threshold:
#         return 1
#     else:
#         return 0






def main2():
    graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5), ('E', 1)],
    'C': [('D', 1), ('E', 1)],
    'D': [('E', 2)],
    'E': []
    }


    y = {
    'AB': 1,
    'AC': 1,  # blocked
    'BC': 1,  # blocked
    'BD': 1,
    'BE': 0,  # blocked
    'CD': 1,
    'CE': 1,
    'DE': 1
    }


    graph = revise_graph_by_y(graph, y)
    source = 'A'
    # dist, prev = dijkstra(graph, source)
    # print(" graph is ", graph)
    # print("Shortest distances from A to all nodes:", find_cost_od(graph, find_all_nodes(graph), source, y))
    # print(dijkstra(graph, source, y))
    # print("Shortest distances from A to E", find_shortest_route(graph, 'A', 'E', y))

    # if 'BC' in find_links_in_shortest_path(graph, 'A', 'D'):
    #     print("hi hi hi")
    # print("Predecessor tree:", prev)
    # find_h_pi(graph, 'A', 'D')
    # print(" links are ", len( 
    # (graph['A']['B'], 'A', 'D')))


if __name__ == '__main__':
    main2()
