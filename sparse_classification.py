import cvxpy as cp
import numpy as np
import time
import scipy 
from sklearn.linear_model import LogisticRegression
from gurobipy import *
#from mosek.fusion import *
from gurobipy import GRB
import gurobipy

def h_ell_cvxpy(y,alpha):
    # Custom loss function
    # h_ell(y, alpha) = y * alpha
    # subject to yα in [-1,0]
    
    term1 = cp.multiply(y, alpha)
    constraints = [term1 <= 0, term1 >= -1]
    return term1, constraints

def valid_cut(X, y, gamma, s):
    # Input:
    #   X: N x D data matrix
    #   y: N-dimensional label vector with entries in {0,1}
    #   gamma: positive scalar
    #   s: sparsity level (number of non-zero entries in beta), size: D
    # Output:
    #   alpha: D-dimensional vector for optimization
    #   maximize -\sum_{i=1}^n h_ell(y_i, x_i^T alpha) - (gamma/2)\sum_{j=1}^D s_j * \alpha.T X_jX_j^T \alpha
    N, D = X.shape
    s = np.array(s)
    # Convert y from {0,1} to {-1,1}
    y    = 2 * y - 1

    s_X = X * s.reshape([1,-1])
    s_X_XT = s_X @ X.T
    s_X_XT = s_X_XT * gamma/2
    # Define variable
    alpha = cp.Variable(N)
    # Define objective
    obj1, con1 = h_ell_cvxpy(y, alpha)
    con1 += [cp.sum(alpha) == 0]

    
    loss = cp.sum(obj1)# + (gamma/2) * cp.quad_form(alpha, np.diag(s) @ (X.T @ X) @ np.diag(s))
    loss += cp.quad_form(alpha, s_X_XT)
    objective = cp.Maximize(-loss)
    problem = cp.Problem(objective, con1)
    problem.solve()


    XTalpha = X.T @ alpha.value
    grad = -gamma/2 * (XTalpha ** 2)
    obj = problem.value

    return grad, obj

def cvx_relaxation(Data, Label, gamma, d):
    N, D = Data.shape
    # Convert y from {0,1} to {-1,1}
    y    = 2 * Label - 1

    # Define variables
    w = cp.Variable(D)
    b = cp.Variable()

    y_pred = Data @ w + b
    loss = cp.pos(1 - cp.multiply(y, y_pred))
    reg = 1/(2 * gamma) * cp.sum_squares(w)
    objective = cp.Minimize(cp.sum(loss) + reg)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.GUROBI, MIPGap=1e-4, TimeLimit=600)
    #print("Status: ", problem.status)
    #print("Optimal value: ", problem.value)
    #print("w: ", w.value)
    #print("b: ", b.value)
    w_val = w.value
    idx = np.argsort(-np.abs(w_val))[:d]
    q = np.zeros(D)
    q[idx] = 1
    return q



def classification_variable_selection(Data, Label, gamma, d, silent=True):
    N, D = Data.shape

    def lazycuts(model, where):
        if where == GRB.Callback.MIPSOL:
            y = model.cbGetSolution(s_var)
            yy = [0] * D
            for i in range(D):
                if y[i] > 0.5:
                    yy[i] = 1
                else:
                    yy[i] = 0
            
            mu, nu = valid_cut(X=Data, y=Label, gamma=gamma, s=yy)

            expr = eta_var 
            rhs  = nu 
            for j in range(D):
                expr -= mu[j] * (s_var[j] - yy[j])
            model.cbLazy(expr,GRB.GREATER_EQUAL,rhs)

    # set model
    m = gurobipy.Model("sparse_classification")
    # create variables
    s_var   = m.addVars(D, vtype=GRB.BINARY, name="s", lb=0, ub=1)
    eta_var = m.addVar(vtype=GRB.CONTINUOUS, name="eta", lb=0, obj=0)
    # set objective
    m.setObjective(eta_var, GRB.MINIMIZE)
    # add constraints
    m.addConstr(s_var.sum() <= d)

    # s_hat = np.zeros(D)
    # #idx = np.random.choice(D, d, replace=False) # randomly chose d indexes from s_hat and set it to be 1
    # s_hat[:d] = 1
    s_hat = cvx_relaxation(Data, Label, gamma, d)

    mu, nu = valid_cut(X=Data, y=Label, gamma=gamma, s=s_hat)
    m.addConstr(eta_var >= nu + sum(mu[j] * (s_var[j] - s_hat[j]) for j in range(D)))
    m.params.OutputFlag = 0
    m.optimize()
    if not silent:
        print("=="*50)
        print("initial solution: ", m.ObjVal)
        print("=="*50)



    s_sol = np.zeros(D)
    for j in range(D):
        s_sol[j] = s_var[j].x
    if not silent:
        print(s_sol)
        print("Objective value: ", m.ObjVal, "upper bound: ", eta_var.x)
    iter = 0
    while (iter <= 100 and eta_var.x - m.ObjVal >= 1e-4):
        iter += 1
        mu, nu = valid_cut(X=Data, y=Label, gamma=gamma, s=s_sol)
        m.addConstr(eta_var >= nu + sum(mu[j] * (s_var[j] - s_sol[j]) for j in range(D)))
        #objval = nu + sum(mu[j] * (s_var[j] - s_sol[j]) for j in range(D))
        m.params.OutputFlag = 0
        m.optimize()
        s_sol = np.zeros(D)
        for j in range(D):
            s_sol[j] = s_var[j].x
    
    for i in range(D):
        s_var[i].vtype = GRB.BINARY
        s_var[i].start = s_hat[i]

    
    m.Params.MIPGap = 1e-4
    m.Params.LazyConstraints = 1
    if not silent:
        m.Params.OutputFlag = 1
    else:
        m.Params.OutputFlag = 0
    m.Params.timelimit = 600
    m.optimize(lazycuts)


    s_sol = np.zeros(D)
    for j in range(D):
        s_sol[j] = s_var[j].x
    
    return s_sol

def classification_cvxpy(Data, Label, gamma, d):
    w = cp.Variable(Data.shape[1])
    b = cp.Variable()
    q = cp.Variable(Data.shape[1], boolean=True)

    y_pred = Data @ w + b
    loss = cp.pos(1 - cp.multiply((2*Label-1), y_pred))
    reg = 1/(2 * gamma) * cp.sum_squares(w)
    objective = cp.Minimize(cp.sum(loss) + reg)
    constr = [cp.sum(q) <= d]
    for i in range(Data.shape[1]):
        constr += [cp.abs(w[i]) <= q[i] * 100]
    problem = cp.Problem(objective, constr)
    problem.solve(solver=cp.GUROBI, MIPGap=1e-4, TimeLimit=600)
    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    print("w: ", w.value)
    print("b: ", b.value)
    print("q: ", q.value)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    s_sol = classification_variable_selection(X, y, gamma=0.1, d=3, silent=False)
    print("Selected variables: ", s_sol)



