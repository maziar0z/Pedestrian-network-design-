import gurobipy as gp
from gurobipy import GRB
from Shortest_path_ import *


# solving the master problem using Gurobi

def solve_master(links, budget, tot_budget, mu, gamma, od_pairs, candidates):

    """
    inputs: links: a list of links in the network
            budget: the budget list for links
            tot_budget: the total budget for the network 
            mu : a dictionary of dual variables for each link and each OD pair
            gamma: a dictionary of dual variables for each OD pair
    """

    # define the model
    model = gp.Model("master_problem")
    model.Params.OutputFlag = 0

    model.setParam('OutputFlag', 0)


    # 1) define dual variables
    # 1.1) Create Z variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    # 1.2) Create y variable
    y = model.addVars(candidates, vtype=GRB.BINARY, name="y")

    # 2) Define the objective function
    model.setObjective(Z , GRB.MAXIMIZE)


#     # 2.1) Build the sum ∑_ℓ mu[ℓ] * y_init[ℓ]
#     con1_sum = gp.quicksum(y[link] * budget[link] for link in links)



    # 3) Add constraints
    # 3.1) Add the constraint for the sum of mu
    model.addConstr(gp.quicksum(y[link] * budget[link] for link in candidates) <= tot_budget  , "c1_budget")

    # 3.2) Add the constraint for the sum over all links and OD pairs
    # model.addConstr(gp.quicksum(gamma[od] - mu[od][link] * y[link] for od in od_pairs for link in candidates) >= 0 , "c2_dual")

    # 3.3) Add the constraint for the nonnegativity of y
    model.addConstr(
        gp.quicksum(gamma[od] for od in od_pairs) + 
        gp.quicksum(mu[od][link] * y[link] for od in od_pairs for link in candidates)
        >= Z,
        "c3_dual"
    )


    # 4) Optimize the model
    model.optimize()


    # 5) Extract results after solving
    if model.Status == GRB.OPTIMAL:
        # Get scalar variable Z
        z_value = Z.X

        # Get decision vector y as a dict {link: value}
        y_values = {link: int(round(y[link].X)) for link in y}
        # print( " In the master SOLVER, objective value  " , model.ObjVal , " y s are "   , y_values.values())
        # print( " mu is " , mu)
        # print(" od pairs are ", od_pairs)
        # print(" heeyy sum is " , sum(gamma.values()))
        # print("OD Pairs:", od_pairs)
        # print("Candidate links:", candidates)
        # print("Sum over OD only:", sum(gamma[od] for od in od_pairs))
        # print(" Sum over mu and y part", sum(mu[od][link] * y_values[link] for od in od_pairs for link in candidates) )
        # print("Sum with nested loop (likely the issue):", sum(gamma[od] for od in od_pairs for link in candidates))

        return model.ObjVal, y_values
    else:
        print("No optimal solution found. Status code:", model.Status)
        return None, None