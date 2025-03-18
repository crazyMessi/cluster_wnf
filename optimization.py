import gurobipy as gp
import numpy as np

def wn2G(wns):
    '''
    
    '''
    pass

'''
找一个目标函数 使得GT情况下的值是最优的
A: w[i,j,1]
B: w[i,j,0]
当label[i] == label[j]时,w[i,j] = w[i,j,1]
当label[i] != label[j]时,w[i,j] = w[i,j,0]
'''
def MIQP(A,B,has_edge,gt=None):
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]
    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    n = len(A)
    x = m.addVars(n, vtype=gp.GRB.BINARY, name="x")
    # Set objective
    obj = gp.QuadExpr()
    obj -= cal_loss(x,A,B,has_edge)
    if gt is not None:
        gt_loss = cal_loss(gt,A,B,has_edge)
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    
    # find the optimal solution
    m.optimize()
    res = np.zeros(n)
    
    # print('Obj: %g' % m.objVal)
    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))
    # print('Optimal solution found')   

    print('Obj: %g' % m.objVal)
    if gt is not None:
        print('gt_loss: %g' % gt_loss)
    for i in range(n):
        res[i] = x[i].x
    return res


def cal_loss(x,A,B,has_edge):
    n = len(x)
    assert A.shape == (n,n)
    assert B.shape == (n,n)
    obj = 0
    for i in range(n):
        for j in range(n):
            if has_edge[i,j]:
                obj += A[i,j]*(1 - (x[i]-x[j])*(x[i]-x[j])) + B[i,j]*(x[i]- x[j])*(x[i] - x[j])
    return obj

def getAB(segment_mean):
    n = len(segment_mean)
    A = np.zeros((n,n))
    B = np.zeros((n,n))
    A = np.abs(segment_mean[None,:] - segment_mean[:,None])
    B = np.abs(segment_mean[None,:] + segment_mean[:,None])
    return A,B


def getAB2(segment_mean):
    n = len(segment_mean)
    A = np.zeros((n,n))
    B = np.zeros((n,n))
    A = np.abs(segment_mean[None,:] - segment_mean[:,None])
    B = np.abs(1 - abs(segment_mean[None,:] -segment_mean[:,None]))
    return A,B
