from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

def CostPerformanceRate(data, xInput, yInput, zInput):

    data = numpy.array(data)
    X, Z = data[:, :2], data[:, 2]

    model = LinearRegression().fit(PolynomialFeatures(degree=3).fit_transform(X), Z)

    z=model.predict(PolynomialFeatures(degree=3).fit_transform([[xInput, yInput]]))[0]
    rate=zInput/z
    return rate

def drawCostPerformance(data):
    #=========
    degree=3
    resolution=200
    #=========

    data = numpy.array(data)
    X = data[:, :2]
    Z = data[:, 2]

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, Z)

    x_range = numpy.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = numpy.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    X_grid, Y_grid = numpy.meshgrid(x_range, y_range)
    Z_grid = model.predict(poly.transform(numpy.c_[X_grid.ravel(), Y_grid.ravel()]))
    Z_grid = Z_grid.reshape(X_grid.shape)

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Z, color='r')
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Price(millian)')
    matplotlib.pyplot.title('PriceMap')
    matplotlib.pyplot.show()
    


#test data
data=[
    (239, 837, 6),
    (450, 182, 3),
    (765, 234, 1),
    (912, 653, 5),
    (374, 509, 10),
    (688, 412, 9),
    (823, 99, 4),
    (123, 456, 2),
    (564, 785, 8),
    (295, 631, 7),
    (447, 782, 1),
    (829, 234, 6),
    (314, 508, 3),
    (890, 100, 10),
    (256, 764, 5),
    (391, 654, 9),
    (472, 812, 2),
    (586, 390, 4),
    (635, 742, 8),
    (178, 568, 7),
    (246, 873, 10),
    (812, 200, 6),
    (193, 654, 3),
    (453, 783, 1),
    (634, 210, 5),
    (799, 512, 4),
    (278, 889, 9),
    (105, 678, 2),
    (472, 134, 8),
    (845, 294, 7),
    (199, 739, 10),
    (981, 657, 6),
    (365, 492, 3),
    (512, 276, 1),
    (729, 835, 5),
    (843, 90, 9),
    (677, 311, 4),
    (890, 486, 2),
    (210, 756, 8),
    (493, 641, 7),
    (365, 543, 10),
    (824, 910, 6),
    (458, 231, 3),
    (620, 874, 1),
    (739, 305, 5),
    (296, 719, 9),
    (152, 870, 4),
    (802, 148, 2),
    (439, 527, 8),
    (275, 640, 7),
    (895, 213, 10),
    (123, 892, 6),
    (744, 634, 3),
    (586, 200, 1),
    (301, 829, 5),
    (970, 452, 4),
    (159, 300, 9),
    (792, 118, 2),
    (284, 463, 8),
    (367, 759, 7),
    (919, 578, 10),
    (441, 135, 6),
    (637, 294, 3),
    (782, 580, 1),
    (554, 471, 5),
    (701, 367, 9),
    (845, 231, 4),
    (290, 896, 2),
    (410, 635, 8),
    (591, 248, 7),
    (322, 147, 10),
    (760, 487, 6),
    (589, 321, 3),
    (940, 504, 1)
]
CPrate=CostPerformanceRate(data, 500, 500, 5)
print(CPrate)

drawCostPerformance(data)
