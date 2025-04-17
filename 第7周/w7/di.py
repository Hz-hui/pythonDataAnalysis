class Array:
    def __init__(self,m,n):
        self._m = m
        self._n = n
        self._matrix = [[0 for j in range(self._n)] for i in range(self._m)]
    def get_shape(self):
        return (self._m, self._n)
    def __getitem__(self, t):
        i, j = t
        return self._matrix[i][j]
    def __setitem__(self, t, v):
        i, j = t 
        self._matrix[i][j] = v

a = Array(100,100)
print(a.get_shape())
a[0,0] = 100
print(a[0,0])
