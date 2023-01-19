>>> x
array([[1],
       [2]])
>>> b
2
>>> A @ x
array([[1]])
>>> A * x
array([[-1,  1],
       [-2,  2]])
>>> A * x +b
array([[1, 3],
       [0, 4]])
>>> A @ x +b
array([[3]])
>>> A @ x + b
array([[3]])
>>> x = np.array([[1,2],[-1,1]]).T
>>> x
array([[ 1, -1],
       [ 2,  1]])
>>> A @ x + b
array([[3, 4]])
>>> x = np.array([[1,2],[-1,1],[1,-1]]).T
>>> A @x + b
array([[3, 4, 0]])
>>> A = np.array([[-1,1,1]])
>>> x = np.array([[1,2,1],[-1,1,1],[1,-1,1]]).T
>>> A = np.array([[-1,1,2]])
>>> A @x 
array([[3, 4, 0]])
>>> x = np.array([[0,0,1],[-1,1,1],[1,-1,1],[1.5,0,1],[0,1.5,1]]).T
>>> A @ x
array([[2. , 4. , 0. , 0.5, 3.5]])
>>> x = np.array([[0,0,1],[-1,1,1],[1,-1,1],[1.5,0,1],[0,-1.5,1]]).T
>>> A @ b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: matmul: Input operand 1 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)
>>> del a
>>> A @ x
array([[2. , 4. , 0. , 0.5, 0.5]])