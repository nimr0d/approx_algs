Run with Python3

Requires numpy, scipy, networkx

### Bin packing:
  Run with 'python3 -i bin_packing.py'
  
  Create a sorted one dimensional numpy array and call bin_pack with some
  epsilon. For example:
  
    >>> items = np.arange(.1, .8, .2)
    
    >>> packing = bin_pack(items, eps=.3)
    
    >>> print(packing)
    
    >>> print(packing * items)

### Unrelated parallel machines:

  Run with 'python3 -i upm.py'
  
  Create a two-dimensional numpy matrix p and call upm. For example:
  
    >>> p = np.ones((4, 5))
    
    >>> x, makespan = upm(p)
    
    >>> print(x)
    
    >>> print(x * p)
    
    >>> print(makespan)https://github.com/nimr0d/approx_algs.git
