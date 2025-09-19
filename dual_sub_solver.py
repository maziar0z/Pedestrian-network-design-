# SOLVES THE DUAL OF SUBPROBLEM, but we do not need it as solutiton of the dual is avaialbe from strong duality theorem


import gurobipy as gp
from gurobipy import GRB
from Shortest_path_ import *

# try:
#     # Print out the Gurobi version string
#     print("Gurobi version:", gp.gurobi.version())

# except gp.GurobiError as e:
#     print("Gurobi import failed:", e)




model1 = gp.Model("int_ex")

# variables
x = model1.addVar(name="box1", vtype=gp.GRB.CONTINUOUS)
y = model1.addVar(name="box2", vtype=gp.GRB.INTEGER)

# objective
model1.setObjective( x + 3 * y, gp.GRB.MINIMIZE)

# constraints
model1.addConstr(x + 5*y >= 5, name="c1")
model1.addConstr( -x + y >= 0, name="c2")
model1.addConstr( y >= 0, name="nonneg")
model1.addConstr( x >= 0, name="nonneg_x")

#solve
model1.optimize()
# print results

print("Objective value:", model1.ObjVal)
print("Box 1:", model1.getVarByName("box1").x)
print("Box 2:", model1.getVarByName("box2").x)



# solve the dual problem using Gurobi
def subproblem_dual(y_init):
    """" for the dual problem, this is for every OD pair
    We go through the sum over all links l avaialble in the target OD pair
    """



class dual_optsolver():
    def ___init__(self, origin, destination):
        """  
        Define graph
        initiate y, h
        """
        self.graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
        }

        self.origin = origin
        self.destination = destination
        # find all links and nodes for the origin-destination pair    
        self.links = find_all_links(self.graph, self.origin, self.destination)   
        self.nodes = find_all_nodes(self.graph, self.origin, self.destination)  
        # for link in self.links:
        #     y[link] = 0

        # self.dual_vars = None

    def solve(self, y_init):

        # define the model
        model = gp.Model("dual_subproblem")

        # 1) define dual variables
        # 1.1) Create mu variables for each link avaiable in this od
        mu = model.addVars(range(1, len(self.links) ), vtype=GRB.CONTINUOUS, name="x")

        # 1.2) Create your other variable y
        gamma = model.addVar(vtype=GRB.INTEGER, name="gamma")

        # 1.3) Build the sum ∑_ℓ mu[ℓ] * y_init[ℓ]
        mu_sum = gp.quicksum(mu[link] * y_init[link] for link in self.links)

        # 2) Define the objective function
        model.setObjective(mu_sum + gamma, GRB.MINIMIZE)

        # 3) Add constraints
        # 3.1) Add the constraint for the sum of mu
        model.addConstr(gp.quicksum(gamma + mu[link] for link in self.links) >= ff , "c1")

        # 3.2) Add the constraint for nonnegativity
        model.addConstr(gamma >= 0, "nonneg_gamma")
        for link in self.links:
            model.addConstr(mu[link] >= 0, f"nonneg_mu_{link}")