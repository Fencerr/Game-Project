import numpy as np
import matplotlib.pyplot as plt


# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


class Curve:
    def __init__(self, controlPoints, intermediatePoints, autoSolve=True):
        # Solution specifications
        self.controlPoints = controlPoints
        self.intermediatePoints = intermediatePoints

        self.solution = None
        self.entityPositions = {}

        # Handle auto-solving
        if autoSolve:
            self.solution = self.evaluateBezier()

    def getBezierCoef(self):
        n = len(self.controlPoints) - 1

        # build coefficients matrix
        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2

        # build points vector
        P = [2 * (2 * self.controlPoints[i] + self.controlPoints[i + 1]) for i in range(n)]
        P[0] = self.controlPoints[0] + 2 * self.controlPoints[1]
        P[n - 1] = 8 * self.controlPoints[n - 1] + self.controlPoints[n]

        # solve system, find a & b
        A = np.linalg.solve(C, P)
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * self.controlPoints[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + self.controlPoints[n]) / 2

        return A, B

    def getBezierCubic(self):
        A, B = self.getBezierCoef()
        return [
            get_cubic(self.controlPoints[i], A[i], B[i], self.controlPoints[i + 1])
            for i in range(len(self.controlPoints) - 1)
        ]

    def evaluateBezier(self):
        curves = self.getBezierCubic()
        return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, self.intermediatePoints)])

    def getSolution(self):
        # Check if the solution was solved already
        if self.solution is None:
            return self.evaluateBezier()

        return self.solution

    def stepEntity(self, E):
        # Check to make sure the surve was solved.
        if self.solution is None:
            raise Exception("Curve was never solved")
            return

        # Check if the entity is in the dictionary.
        if E not in self.entityPositions:
            self.entityPositions[E] = 0
            return self.solution[0]

        # Make sure the index is within the bounds of the solution
        if self.entityPositions[E] >= len(self.solution) - 1:
            return None

        self.entityPositions[E] += 1
        return self.solution[self.entityPositions[E]]

    def graph(self):
        # Verify that the solution has been solved
        if self.solution is None:
            raise Exception("Curve was never solved")
            return

        # Get point coordinates
        x, y = self.controlPoints[:, 0], self.controlPoints[:, 1]

        # Get solution coordinates
        px, py = self.solution[:, 0], self.solution[:, 1]

        # Draw
        plt.figure(figsize=(11, 8))
        plt.plot(px, py, 'b-')
        plt.plot(x, y, 'ro')
        plt.show()


class Entity:
    pass


# generate 5 random points
points = np.random.rand(23, 2)

# create the curve
c = Curve(points, 50)
c.graph()

Running = True
E = Entity()
while Running:
    position = c.stepEntity(Entity)
    print(position)
    if position is None:
        Running = False
