import networkx as nx
import numpy as np 
import scipy as sp
from scipy.optimize import linprog

# Get constraints that depend on t
def constraints(p, t):
  m, n = p.shape
  nz = np.nonzero(p.flatten() > t)
  c = len(nz[0])
  A_eq = np.zeros((c, m * n))
  A_eq[np.arange(c), nz] = 1
  b_eq = np.zeros(c)
  b_ub = np.ones(m) * t
  return A_eq, b_eq, b_ub

# Solve relaxed LP for a given t
def upm_t(p, t, c):
  m, n = p.shape
  A_eq, b_eq, A_ub = c
  A_eq_, b_eq_, b_ub = constraints(p, t)
  A_eq = np.concatenate((A_eq, A_eq_))
  b_eq = np.concatenate((b_eq, b_eq_))
  obj = np.zeros(m * n)
  return linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

# Deterministic rounding of LP
def rnd(x):
  m, n = x.shape
  x_r = x.round(decimals=10) # To fix some floating point issues
  # Assign jobs which are already assigned by the LP
  x_ = (x_r >= 1)
  # Create a graph from the rest of the nodes
  s = sp.sparse.csr_matrix((x_r - x_) > 0)
  G = nx.algorithms.bipartite.from_biadjacency_matrix(s)
  for cycle in nx.cycle_basis(G):
    # Assign first job in cycle arbitrarily to an adjacent machine and remove it
    machine, job = sorted(cycle[:2])
    x_[machine, job - m] = 1
    G.remove_node(job)
    successors = nx.dfs_successors(G, source=machine)
    # Assign the rest of the jobs in the connected component down in the DFS tree
    for k, v in successors.items():
      # Check if node is a job node
      if k >= m:
        x_[v[0], k - m] = 1
  return x_

# Unrelated parallel machines solver
def upm(p):
  """
  :param p: m x n numpy array
  """
  m, n = p.shape
  # Constraints that do not depend on t
  A_eq = (np.ones((n, m, 1)) * np.eye(n)[:, None, :]).reshape((-1, m * n))
  b_eq = np.ones(n)
  A_ub = (p * np.eye(m)[..., None]).reshape((-1, m * n))
  c = A_eq, b_eq, A_ub
  # Do greedy algorithm to get upper and lower bounds
  mp = np.ma.array(p, mask=True)
  mp.mask[p.argmin(axis=0), np.arange(n)] = False
  # Binary search for t
  u = mp.sum(axis=1).max()
  l = u // m
  best_x = np.logical_not(mp.mask)
  while u > l:
    d = (u + l) // 2
    res = upm_t(p, d, c)
    if res.success:
      u = d
      best_x = rnd(res.x.reshape((m, n)))
    else:
      l = d + 1
  return best_x, (p * best_x).sum(axis=1).max()
