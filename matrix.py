from local_maths import *

class Matrix:
    rows = 0
    columns = 0
    matrx = [[]]

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.matrx = [[0 for col in range(columns)] for row in range(rows)]

    @staticmethod
    def columnMatrixFromArray(array):
        m = Matrix(len(array), 1)

        for i in range(len(array)):
            m.matrx[i][0] = array[i]

        return m

    def multiplyByMatrix(self, m):
        r = Matrix(self.rows, m.columns)

        if self.columns == m.rows:
            for i in range(self.rows):
                for j in range(m.columns):
                    sum = 0
                    for k in range(self.columns):
                        sum += self.matrx[i][k] * m.matrx[k][j]
                    r.matrx[i][j] = sum
        
        return r

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.columns):
                self.matrx[i][j] = 1 + (np.random.rand() * -2.0)

    def addBias(self):
        m = Matrix(self.rows+1, 1)

        for i in range(self.rows):
            m.matrx[i][0] = self.matrx[i][0]
        
        m.matrx[self.rows][0] = 1
        self.rows += 1
        self.matrx = m.matrx

    def activate(self):
        for i in range(self.rows):
            for j in range(self.columns):
                self.matrx[i][j] = sigmoid(self.matrx[i][j])

    def mutate(self, mutation_rate):
        for i in range(self.rows):
            for j in range(self.columns):
                r = np.random.rand()
                if r < mutation_rate:
                    self.matrx[i][j] += np.random.normal()
                    if self.matrx[i][j] > 1:
                        self.matrx[i][j] = 1
                    if self.matrx[i][j] < -1:
                        self.matrx[i][j] = -1

    def clone(self):
        c = Matrix(self.rows,self.columns)

        for i in range(self.rows):
            for j in range(self.columns):
                c.matrx = self.matrx[i][j]

        return c

    def toArray(self):
        arr = []

        for i in range(self.rows):
            for j in range(self.columns):
                arr.append(self.matrx[i][j])
                
        return arr
    
    def print(self):
        for i in range(self.rows):
                print(self.matrx[i])    