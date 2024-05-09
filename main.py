from ExpressionTree import *
import math
import matplotlib.pyplot as plt

tree = ExpressionTree(
    ["x"],
    [
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("/", lambda x, y: x / y),
        ("*", lambda x, y: x * y)
    ],
    [
        ("sin", math.sin),
        ("cos", math.cos)
    ],
    (-3, 3),
    True,
    0.1,
    1,
    0.9,
    1000
)


def GenerateRandomPopulation(configs, treeSetup):
    population = []
    for degeradationRate, count in configs:
        for _ in range(count):
            tree = copy.deepcopy(treeSetup)
            tree.MakeRandom(0.999, degeradationRate)
            population.append(tree)
    return population

xrange = (-10, 10)
yrange = (-500000, 500000)
function = lambda x: (26 + x) * x * -1000
datapoints = np.linspace(-10, 10, 10)
target = [({"x": x}, function(x)) for x in datapoints]

tree.MakeRandom(0.999, 0.8)

# tree.root = BinaryOperandNode(None, lambda x, y: x * y, "*")
# tree.root.right = StaticNode(tree.root, 1000)
# tree.root.left = BinaryOperandNode(tree.root, lambda x, y: x * y, "*")
# tree.root.left.right = VariableNode(tree.root.left, "x")
# tree.root.left.left = BinaryOperandNode(tree.root, lambda x, y: x + y, "+")
# tree.root.left.left.left = VariableNode(tree.root.left.left, "x")
# tree.root.left.left.right = StaticNode(tree.root.left.left, -400)
# tree.nodes = []
# tree.nodes.append(tree.root)
# tree.nodes.append(tree.root.right)
# tree.nodes.append(tree.root.left)
# tree.nodes.append(tree.root.left.right)
# tree.nodes.append(tree.root.left.left)
# tree.nodes.append(tree.root.left.left.left)
# tree.nodes.append(tree.root.left.left.right)

print(tree.display())
valuesBeforeOptim = [tree.evaluate({"x": x}) for x in datapoints]
tree.optimizeStatic(target)
valuesAfterOptim = [tree.evaluate({"x": x}) for x in datapoints]
print(tree.display())
# population = GenerateRandomPopulation([(0.8, 10)], tree)
# for tree in population:
#     print(tree.display())
# DataPoints = np.linspace(xrange[0], xrange[1], (xrange[1] - xrange[0]) * 10)
correct = [function(x) for x in datapoints]
print(correct)
plt.ylim(yrange[0], yrange[1])
plt.xlim(xrange[0], xrange[1])
plt.plot(datapoints, valuesBeforeOptim, color="red")
plt.plot(datapoints, valuesAfterOptim, color="green")
plt.plot(datapoints, correct, color="blue")
plt.show()