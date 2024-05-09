from ExpressionTree import *
import math
import matplotlib.pyplot as plt
from queue import PriorityQueue
import copy
import matplotlib.axes as ax
from matplotlib.axes import Axes
import time


tree = ExpressionTree(
    ["x"],
    [
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("/", lambda x, y: x / y),
        ("*", lambda x, y: x * y),
    ],
    [
        ("sin", math.sin),
        ("cos", math.cos)
    ],
    (-3, 3),
    True,
    0.1,
    1,
    0.8,
    300
)
tree2 = ExpressionTree(
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
    0.95,
    300
)

def GenerateRandomPopulation(configs, treeSetup):
    population = []
    for degeradationRate, count in configs:
        for _ in range(count):
            while True:
                tree = copy.deepcopy(treeSetup)
                tree.MakeRandom(0.999, degeradationRate)
                if type(tree.root) != StaticNode:
                    break
            population.append(tree)
    return population


def reproduce(population, mutationChance):
    children = set()
    #print(len(population))
    i = 0
    for parent1 in population:
        for parent2 in population:
            if parent1 != parent2:
                #print(i)
                i += 1
                a, b = ExpressionTree.crossover(parent1, parent2)
                chance = np.random.uniform(0, 1)
                if chance < mutationChance:
                    a.Evolve()
                children.add(a)
                chance = np.random.uniform(0, 1)
                if chance < mutationChance:
                    b.Evolve()
                children.add(b)
    return children

def K_Dynamic(candidates, k):
    print("here")
    total = 0
    for candidate in candidates:
        if candidate[0] != -math.inf and candidate[0] != math.nan:
            total += -candidate[0] / 10000
        else:
            candidates.remove(candidate)
    print(total)
    prob = 0
    probs = []
    population = set()
    for candidate in candidates:
        probs.append(prob)
        prob += (-candidate[0] / 10000) / total
    while len(population) != k:
        choice = np.random.uniform(0, 1)
        #print(item)
        for i in range(len(probs)):
            if probs[i] > choice:
                population.add(candidates[i])
                break
    #print(queue.queue)
    return population

def K_Best(candidates, k):
    queue = PriorityQueue(k)
    for item in candidates:
        try:
            #print(item)
            if queue.qsize() < k:
                queue.put(item)
                continue
            max = queue._get()
            #print(max)
            best = item if item[0] > max[0] else max
            queue.put(best)
        except:
            continue
    #print(queue.queue)
    return queue.queue

def Train(initialPopulation, target, xrange, yrange, k, mutationChance, DataPoints, correct):
    fig, axes = plt.subplots(1, 1)
    axes: Axes = axes
    axes.set_xlim(xrange[0], xrange[1])
    axes.set_ylim(yrange[0], yrange[1])
    text = axes.text(0.03, 0.97, "", transform=axes.transAxes, va="top")
    correctFunction, = axes.plot(DataPoints, correct, color="blue")
    learnedFunction, = axes.plot([], [], color="red")
    lastupdate = time.time()
    plt.show(block=False)
    population = initialPopulation
    changeCounter = 50
    globalBestTree = None
    globalBestLoss = math.inf
    genNumber = 0
    while True:
        genNumber += 1
        candidates = []
        i = 0
        print(f"Generation: {genNumber}")
        print("Proccessing Initial Population...")
        for tree in population:
            i += 1
            try:
                tree.optimizeStatic(target)
                loss = tree.SqueredLoss(target)
                #print(i, loss)
                #print(loss, tree.display())
                candidates.append((-loss, tree))
            except:
                None
        print("Deciding Winners...")
        #winners = K_Dynamic(candidates, k) if changeCounter <= 30 else K_Best(candidates, k)
        winners = K_Best(candidates, k)
        bestTree = None
        bestloss = math.inf
        temp = set()
        j = 0
        for loss, tree in winners:
            #print(j)
            j += 1
            #tree.simplify(tree.root)
            #temp = set()
            #tree.nodes = ExpressionTree.gatherNodes(tree)
            #tree.optimizeStatic(target)
            loss = tree.SqueredLoss(target)
            if loss < bestloss:
                bestTree = tree
                bestloss = loss
            temp.add(tree)
        #text.set_text(f"epoch: {epoch}\nbatch: {batch}\nloss: {loss}")
        print("Best Loss in Winners:", bestloss)
        if globalBestLoss > bestloss:
            globalBestLoss = bestloss
            bestTree.optimizeStatic(target)
            globalBestTree = bestTree
            changeCounter = 30
        print("Best Global Loss:", globalBestLoss)
        if globalBestLoss < 0.01 or changeCounter == 0:
            return globalBestTree
        print("Reproduction...")
        children = reproduce(temp, mutationChance)
        print("Max Depth in children", max([tree.depth for tree in children]))
        print("Max nodeCount in children", max([len(tree.nodes) for tree in children]))
        for item in children:
            if type(item) != float:
                temp.add(item)
        #print(children)
        #print(len(children))
        changeCounter -= 1
        population = list(temp)
        if time.time() - lastupdate > 0.01:
            learnedFunction.set_data(DataPoints, [globalBestTree.evaluate({"x": x}) for x in DataPoints])
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(1e-3)
            lastupdate = time.time()
        print("************************")

    
        
xrange = (-5, 5)
yrange = (-10, 100)
function = lambda x: 20* np.sinc(x)
#function = lambda x: (x + 5) * 2 + x
datapoints = np.linspace(xrange[0], xrange[1], (xrange[1] - xrange[0]) * 5)
target = [({"x": x}, function(x)) for x in datapoints]
# tree1.MakeRandom(0.999, 0.8)
# tree2.MakeRandom(0.999, 0.8)
# print("A: ", tree1.display())
# print()
# print("B: ", tree2.display())
# print()
# a, b = ExpressionTree.crossover(tree1, tree2)
# print("A: ", tree1.display())
# print()
# print("B: ", tree2.display())
# print()
# print("C: ", a.display())
# print()
# print("D: ", b.display())

initialPopulation = GenerateRandomPopulation([(0.6, 200), (0.9, 100), (0.95, 50)], tree)
correct = [function(x) for x in datapoints]
#print([tree.display() for tree in initialPopulation])
ans = Train(initialPopulation, target, xrange, yrange, 50, 0.2, list(datapoints), correct)
print(ans.display())

valuesAfterOptim = [ans.evaluate({"x": x}) for x in datapoints]

#text = plt.text(-9.5, 8, f"x * x + rad(x) + 30\n{ans.display()}")
plt.ylim(yrange[0], yrange[1])
plt.xlim(xrange[0], xrange[1])
plt.plot(datapoints, valuesAfterOptim, color="green")
plt.plot(datapoints, correct, color="blue")
plt.show()