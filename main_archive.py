from Shortest_path_ import *
from master_problem import *
from reader import *

import gurobipy as gp
from gurobipy import GRB

import time

""" 
Solving the main problem:  brings together the dual subproblem solver, the subproblem (the shortest path solver), and the master problem.
"""

# not dijkstra for each od, just for each o and save it
# for f, it f=1, just pass 1
# in dijkstra, if cost <c  make sure to stop right away

# STEP 0: INTITIALIZE

# initalize the network graph, length tresh, and budget 
# Define the network graph


df = read_tntp_network("SiouxFalls_net.tntp")
graph = build_graph(df, weight_col="free_flow_time")

graph_original = graph.copy()

# # define the accessibility of each link



# y_build = {
# 'AB': 0,
# 'AC': 0,  
# 'BC': 0,  
# 'BD': 0,
# 'BE': 0,  
# 'CD': 0,
# 'CE': 0,
# 'DE': 0
# }



# initialize y_init for the dual subproblem
# find all links and nodes for the origin-destination pair
links = find_all_links(graph)
nodes = find_all_nodes(graph)

# print("All links in the network:", links)

y = {}
for link in links:
    y[link] = 1

max_lb = 1

y['1917'] = 0
y['1719'] = 0

y['2019'] = 0
y['1920'] = 0

y['2018'] = 0
y['1820'] = 0

y['1312'] = 0
y['1213'] = 0

y['1423'] = 0
y['2314'] = 0

y['12'] = 0
y['21'] = 0

y_init = y.copy()  # make a copy of y to use as initial values for the dual subproblem

# read the trips data for the demand
demand = read_trips_tntp("SiouxFalls_trips.tntp")

# set the number of agents at each node
o = {}
for node in nodes:
    o[node] = sum(demand[node].values())  # total number of agents at each node

print("sum of the demand is " , sum(o[node] for node in nodes))

# print("Number of agents at each node:", o[org])
# print(" node type is " , type(nodes[0]))

# define the 
c = 6 # length threshold for the path

# define the budget for each link
budget = {}
for link in links:
    budget[link] = 10

w = {}  # weight for each destination, can be changed as needed
for node in nodes:
    w[node] = 1  # weight for each destination, can be changed as needed

# total budget for the network
tot_budget = 65

candidates = []
# find the candidate links for the master problem
for link in links:
    if y[link] == 0: # if the link is not built yet
        candidates.append(link)


# find cost for a given od using dijakstra
def find_cost_od(graph, nodes, origin, y):
# finds the cost from origin to all other nodes
    dist, _ = dijkstra(graph, origin, y)
    
    # Filter for only nodes in the graph to be safe
    cost_dict = {node: dist.get(node, float('inf')) for node in nodes if node != origin}
    return cost_dict



# def find_f_od(graph, origin, destination, threshold, y, od_costs_dict):
#     """
#     Finds the f_od binary value for a given origin and destination in the graph.
    
#     Parameters:
#         graph (dict): Adjacency list representation of the graph.
#         origin (any): Origin node
#         destination (any): Destination node
#         c (int): length threshold for the path
#         od_costs_dict (dict): Dictionary of OD costs

#     Returns:
#         f_od (binary): whether the destination is reachable within the length threshold c.
#     """
#     if od_costs_dict[origin][destination] <= threshold:
#         return 1
#     else:
#         return 0


# # run the Dijkastra's once for the first time, to save od costs
# od_costs = {}
# for node in nodes:
#     od_costs[node] = find_cost_od(graph, nodes, node, y)


od_pairs = get_all_od_pairs_from_trips(demand)

# print("All OD pairs in the network:", od_pairs)

# find origins and destinations given the demand
od_to_origin, od_to_dest = build_od_mappings(demand)

# print("Origin to OD mapping:", od_to_origin)


def add_link_to_graph(original_graph, current_graph, link, y):
    """
    Returns a copy of current_graph with the given link added (if y[link] == 1),
    using the cost from original_graph.
    
    Parameters:
        original_graph: Full graph with all links and costs.
        current_graph: Graph to be updated (dict).
        link: A string like 'CD'.
        y: Dictionary of link status.
    Returns:
        updated_graph: The modified graph with the link added (if applicable).
    """
    from copy import deepcopy
    updated_graph = deepcopy(current_graph)
    
    if y.get(link, 0) == 1:
        from_node = link[0]
        to_node = link[1]
        
        # Search for cost in original graph
        for dest, cost in original_graph.get(from_node, []):
            if dest == to_node:
                # Add if not already there
                if to_node not in [d for d, _ in updated_graph.get(from_node, [])]:
                    updated_graph[from_node].append((to_node, cost))
                break
    return updated_graph



# find the value of the dual variable for a given origin and destination
def daul_value(graph, origin, destination, y_building , links):
    """ 
    This function calculates the dual value for a given origin and destination
    using the initial dual variable y_building.

    inputs: network graph, origin node, destination node, all network links, f(d_pi) as binary dictionary
    """
    # Initialize the shortest path solver
    h_pi = find_h_pi(graph, origin, destination, y_building)
    gamma = demand[origin][destination] * sum(h_pi.values()) * find_f_od(graph, origin, destination, c, y_building)  # Sum of h pi values

    mu = {}
    for link in links:
        # print( "Checking link:", link )
        # print( "and y_init[link]:", y_init[link])
        if y_building[link] == 0: # if not built yet and that link is not already in the original network
            y_building[link] = 1 #    1. set y[link] to 1 in order to check shortest path  
            # print( " for od " , origin, " and destination ", destination, " link is ", link, " we have y = 0, SP is ", find_links_in_shortest_path(graph, origin, destination, y_building))
            if link in find_links_in_shortest_path(graph, origin, destination, y_building):  # 3. check if the link is in the shortest path
                mu[link] = 1 * demand[origin][destination]
                # print( " for od " , origin, " and destination ", destination, " link is ", link, " we can add y =1 to get the shortest path")
            else:
                mu[link] = 0
            y_building[link] = 0
        else:
            mu[link] = 0
    return gamma, mu

origins = build_od_mappings

# STEP 1: FOR EACH OD: find the links, nodes, and the shortest path h_pi (subproblem), 
gamma = {}
mu = {}
h_pi = {}
gap = 100
iteration = 0
obj_value = -10
obj_value_prev = 0

gap_threshold = 0.05  # Define a threshold for the gap to stop the iterations





# Step 2: form Master problem optimization problem for the first time
# Build master problem once
master = gp.Model("master_problem")
master.Params.OutputFlag = 0   # silence logs if you want

# Decision vars for candidate links
y_vars = master.addVars(candidates, vtype=GRB.BINARY, name="y")

# 1.1) Create Z variable
Z = master.addVar(vtype=GRB.CONTINUOUS, name="Z")

# Budget constraint
master.addConstr(gp.quicksum(y_vars[link] * budget[link] for link in candidates) <= tot_budget, "budget")

# Objective placeholder (will be updated with cuts)
master.setObjective(Z, GRB.MAXIMIZE)

master.update()





start_dual = 0
start_master = 0
finish_dual = 0
finish_master = 0
total_dual = 0
total_master = 0


# iterate until the gap is small enough
while gap > gap_threshold:
    objvalue_subproblem = 0
    objvalue_dualsubproblem = 0
    # print( " starting with y_init: ", y_init)

    y_final = y_init.copy()  # make a copy of y_init to use as final values for the dual subproblem

    # # update the network using the new y values found in the master problem
    # for link in candidates:
    #     if y_init[link] == 1 and y[link] == 0:  # if the link is built in the master problem but not in the original network
    #         y[link] = 1  # update the link in the original network
    #         # # update the graph with the new y values
    #         # graph = add_link_to_graph(graph_original, graph, link, y)


    # print("Updated graph with new y values:", graph)

    # run the Dijkstra's once for the first time, to save od costs
    # od_costs = {}
    # for node in nodes:
    #     od_costs[node] = find_cost_od(graph, nodes, node, y)


    start_dual = time.time()


    for od in od_pairs:
        orig = od_to_origin[od]
        dest = od_to_dest[od]
        obj_value_prev = obj_value
        # print("Processing OD pair:",  od, " origin ", orig , "  and ", dest)

        # find h_ pi
        h_pi[od] = find_h_pi(graph, orig, dest, y_init)

        objvalue_subproblem +=  sum(h_pi[od].values()) * find_f_od(graph, orig, dest, c, y_init) * demand[orig][dest] #* o[orig] * w[dest]

        # find dual values 
        gamma[od], mu[od] = daul_value(graph, orig, dest, y_init, links)

        # find the objective value for the dual subproblem
        objvalue_dualsubproblem += (gamma[od] + sum(mu[od][link] * y_init[link] for link in candidates)) #* o[orig] * w[dest]

    finish_dual = time.time()
    total_dual += finish_dual - start_dual
        # if od == '2019':
        #     print(" h pi for OD pair", od, ":", sum(h_pi[od].values()))
        #     print("gamma values for OD pair", od, ":", gamma[od])

    # print( "sum of gamma is " , sum(gamma[od] for od in od_pairs))
    # print(" fixed y values for OD pair", ":", y_fixed)
    # print(" H piii", h_pi)

    print(f"Objective value of the subproblem: {objvalue_subproblem}")
    print(f"Objective value of the dual subproblem: {objvalue_dualsubproblem}")

    # report the objective value of the subproblem and dual of its subproblem
    if objvalue_subproblem != objvalue_dualsubproblem:
        print(" WARNING")
        SystemError
    else:
        lower_bound = objvalue_subproblem # set the LB

    lower_bound = objvalue_subproblem # set the LB

    if lower_bound > max_lb:
        max_lb = lower_bound



    # Update and solve MASTER problem
    # 3.3) Add the constraint for the nonnegativity of y
    start_master = time.time()
    master.addConstr(
        sum(gamma.values()) + 
        gp.quicksum(mu[od][link] * y_vars[link] for od in od_pairs for link in candidates)
        >= Z,
        "c3_dual")    
    

    master.optimize()
    finish_master = time.time()
    total_master += finish_master - start_master
    upper_bound = master.objVal
    y_values = {link: int(round(y_vars[link].X)) for link in candidates}

    # print("Sum over OD only:", sum(gamma[od] for od in od_pairs))
    # print(" Sum over mu and y part", sum(mu[od][link] * y_values[link] for od in od_pairs for link in candidates) )

    # update y_init with solution
    for link in candidates:
        y_init[link] = y_values[link]

    print(" total time for dual subproblem is ", total_dual)
    print(" total time for master subproblem is ", total_master)
    iteration += 1
    gap = abs(max_lb - upper_bound)/max_lb
        
    print(f"Iteration {iteration}: UB = {upper_bound}, LB = {max_lb} ,  Gap = {gap}", " optimization y is", y_values)



print("STOPED, Final Objective Value:", lower_bound, " y_values are: ", y_final)





















