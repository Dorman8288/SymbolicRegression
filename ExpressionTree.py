import numpy as np
import copy 
import math

class ExpressionTree:
    def __init__(self, variables, BinaryOperators, UnaryOperators, staticRange, continuos, optimizationStep, initT, coolingRate, numIterations) -> None:
        self.optimzationStep = optimizationStep
        self.numIterations = numIterations
        self.initTempreture = initT
        self.coolingRate = coolingRate
        self.root = None
        self.variables = variables
        self.BinaryOperators = BinaryOperators
        self.UnaryOperators = UnaryOperators
        self.range = staticRange
        self.continous = continuos
        self.depth = 0 

    def MakeRandom(self, prob, degradationRate):
        self.nodes = []
        self.root = self.Make(prob, degradationRate, 0, self.nodes, None)
        self.simplify(self.root)
    
    def simplify(self, node):
        if type(node) is BinaryOperandNode:
            left = self.simplify(node.left)
            right = self.simplify(node.right)
            if left == None and right == None:
                return None
            elif left == None and right != None:
                self.nodes.remove(node.right)
                node.right = StaticNode(node, right)
                self.nodes.append(node.right)
            elif left != None and right == None:
                self.nodes.remove(node.left)
                node.left = StaticNode(node, left)
                self.nodes.append(node.right)
            else:
                if node.parent == None:
                    self.nodes.remove(self.root)
                    self.root = StaticNode(None, node.operation(left, right))
                    self.nodes.append(self.root)
                return node.operation(left, right)
        elif type(node) is UnaryOperandNode:
            child = self.simplify(node.child)
            if child == None:
                return None
            else:
                if node.parent == None:
                    self.nodes.remove(self.root)
                    self.root = StaticNode(None, node.operation(child))
                    self.nodes.append(self.root)
                return node.operation(child)
        elif type(node) is VariableNode:
            return None
        else:
            return node.value


    def gatherNodes(x, gatheredNodes: set):
        gatheredNodes.add(x)
        if type(x) == BinaryOperandNode:
            ExpressionTree.gatherNodes(x.left, gatheredNodes)
            ExpressionTree.gatherNodes(x.right, gatheredNodes)
        elif type(x) == UnaryOperandNode:
            ExpressionTree.gatherNodes(x.child, gatheredNodes)
        else:
            return

    def MergeNodes(nodes_A, nodes_B, choice_A, choice_B):
        buffer_A = set()
        ExpressionTree.gatherNodes(choice_A, buffer_A)
        buffer_B = set()
        ExpressionTree.gatherNodes(choice_B, buffer_B)
        for node in buffer_A:
            nodes_A.remove(node)
            nodes_B.append(node)
        for node in buffer_B:
            nodes_B.remove(node)
            nodes_A.append(node)


    def crossover(a, b):
        a = copy.deepcopy(a)
        b = copy.deepcopy(b)
        # print(a.display())
        # print(b.display())
        while True:
            choice_A = np.random.randint(0, len(a.nodes))
            choice_B = np.random.randint(0, len(b.nodes))
            node_A = a.nodes[choice_A]
            node_B = b.nodes[choice_B]
            parent_A = node_A.parent
            parent_B = node_B.parent
            if parent_A != None and parent_B != None:
                break
        if type(parent_A) is BinaryOperandNode and type(parent_B) is BinaryOperandNode:
            if parent_A.right == node_A and parent_B.right == node_B:
                parent_A.right, parent_B.right = parent_B.right, parent_A.right
            if parent_A.right == node_A and parent_B.left == node_B:
                parent_A.right, parent_B.left = parent_B.left, parent_A.right
            if parent_A.left == node_A and parent_B.right == node_B:
                parent_A.left, parent_B.right = parent_B.right, parent_A.left
            if parent_A.left == node_A and parent_B.left == node_B:
                parent_A.left, parent_B.left = parent_B.left, parent_A.left
        if type(parent_A) is UnaryOperandNode and type(parent_B) is BinaryOperandNode:
            if parent_B.left == node_B:
                parent_B.left, parent_A.child = parent_A.child, parent_B.left
            if parent_B.right == node_B:
                parent_B.right, parent_A.child = parent_A.child, parent_B.right
        if type(parent_A) is BinaryOperandNode and type(parent_B) is UnaryOperandNode:
            if parent_A.left == node_A:
                parent_A.left, parent_B.child = parent_B.child, parent_A.left
            if parent_A.right == node_A:
                parent_A.right, parent_B.child = parent_B.child, parent_A.right
        if type(parent_A) is UnaryOperandNode and type(parent_B) is UnaryOperandNode:
            parent_A.child, parent_B.child = parent_B.child, parent_A.child
        node_A.parent, node_B.parent = parent_B, parent_A
        # print(a.display())
        # print(b.display())
        temp = set()
        ExpressionTree.gatherNodes(a.root, temp)
        a.nodes = list(temp)
        temp = set()
        ExpressionTree.gatherNodes(b.root, temp)
        b.nodes = list(temp)
        #ExpressionTree.MergeNodes(a.nodes, b.nodes, node_A, node_B)
        return a, b

    def Evolve(self):
        choice = np.random.randint(0, len(self.nodes))
        node = self.nodes[choice]
        if type(node) is StaticNode:
            node.changeValue(range=self.range, continous=self.continous)
        elif type(node) is VariableNode:
            prevName = node.name
            while prevName == node.name and len(self.variables) != 1:
                variable = np.random.randint(0, len(self.variables))
                node.name = self.variables[variable]
        elif type(node) is BinaryOperandNode:
            prevOp = node.symbol
            while prevOp == node.symbol:
                operator = np.random.randint(0, len(self.BinaryOperators))
                node.operation = self.BinaryOperators[operator][1]
                node.symbol = self.BinaryOperators[operator][0]
        elif type(node) is UnaryOperandNode:
            prevOp = node.symbol
            while prevOp == node.symbol:
                operator = np.random.randint(0, len(self.UnaryOperators))
                node.operation = self.UnaryOperators[operator][1]
                node.symbol = self.UnaryOperators[operator][0]

    def Make(self, prob, degradationRate, depth, usedNodes: list, parent):
        self.depth = max(depth, self.depth)
        type = np.random.binomial(1, prob)
        node = None
        if type == 1:
            opType = np.random.randint(0, 2)
            if opType == 0:
                operator = np.random.randint(0, len(self.BinaryOperators))
                node = BinaryOperandNode(parent, self.BinaryOperators[operator][1], self.BinaryOperators[operator][0])
                node.left = self.Make(prob * degradationRate, degradationRate, depth + 1, usedNodes, node)
                node.right = self.Make(prob * degradationRate, degradationRate, depth + 1, usedNodes, node)
            else:
                operator = np.random.randint(0, len(self.UnaryOperators))
                node = UnaryOperandNode(parent, self.UnaryOperators[operator][1], self.UnaryOperators[operator][0])
                node.child = self.Make(prob * degradationRate, degradationRate, depth + 1, usedNodes, node)
        else:
            valueType = np.random.randint(0, 2)
            if valueType == 0:
                variable = np.random.randint(0, len(self.variables))
                node = VariableNode(parent, self.variables[variable])
            else:
                node = StaticNode(parent, range=self.range, continous=self.continous)
        usedNodes.append(node)
        return node
    
    def optimizeStatic(self, target):
        statics = []
        bestAns = []
        bestloss = self.SqueredLoss(target)
        for node in self.nodes:
            if type(node) is StaticNode:
                statics.append(node)
                bestAns.append(node.value)
        #print("StaticCount: ", len(statics))
        if len(statics) == 0:
            return
        #print(self.SqueredLoss(target))
        tempreture = self.initTempreture
        changeCounter = 0
        while True:
            if tempreture < 0.001:
                break
            for i in range(self.numIterations):
                if changeCounter == 100:
                    break
                initialLoss = self.SqueredLoss(target)
                if bestloss > initialLoss:
                    bestloss = initialLoss
                    bestAns = [static.value for static in statics]
                step = self.generateRandomStep(len(statics))
                for i in range(len(statics)):
                    statics[i].value = statics[i].value + step[i]
                SecondaryLoss = self.SqueredLoss(target)
                #print(initialLoss, SecondaryLoss)
                Energy = initialLoss - SecondaryLoss
                #print(initialLoss)
                if Energy < 0:
                    errorProb = math.exp(Energy/tempreture/ 100)
                    #print(errorProb)
                    if errorProb < np.random.uniform(0, 1):
                        for i in range(len(statics)):
                            changeCounter += 1
                            statics[i].value = statics[i].value - step[i]
            tempreture *= self.coolingRate
        #print(bestAns)
        for i in range(len(statics)):
            statics[i].value = bestAns[i]

    def generateRandomStep(self, count):
        vector = []
        for _ in range(count):
            choice = np.random.randint(0, 2)
            vector.append((-1 ^ choice) * self.optimzationStep)
        return vector
        
    def SqueredLoss(self, points):
        loss = 0
        for (variables, value) in points:
            pred = self.root.getValue(variables)
            loss += (pred - value) * (pred - value)
        return loss / len(points)

    def evaluate(self, variableValues):
        try:
            return self.root.getValue(variableValues)
        except:
            return None
    
    def display(self):
        return self.root.display()
    





class StaticNode:
    def __init__(self, parent, value = None, range = None, continous = False):
        super().__init__()
        self.value = value
        self.parent = parent
        if value == None:
            if continous:
                self.value = np.random.uniform(range[0], range[1])
            else:
                self.value = np.random.randint(range[0], range[1])

    def changeValue(self, value = None, range = None, continous = False):
        self.value = value
        if value == None:
            if continous:
                self.value = np.random.uniform(range[0], range[1])
            else:
                self.value = np.random.randint(range[0], range[1])   

    def display(self):
        return str(self.value)

    def getValue(self, variables):
        return self.value

class VariableNode:
    def __init__(self, parent, name):
        super().__init__()
        self.name = name
        self.parent = parent

    def getValue(self, variables):
        return variables[self.name]
    
    def display(self):
        return self.name

class BinaryOperandNode:
    def __init__(self, parent, operation, symbol):
        super().__init__()
        self.right = None
        self.left = None
        self.operation = operation
        self.symbol = symbol
        self.parent = parent

    def getValue(self, variables):
        return self.operation(self.left.getValue(variables), self.right.getValue(variables))

    def display(self):
        return f"({self.left.display()} {self.symbol} {self.right.display()})"


class UnaryOperandNode:
    def __init__(self, parent, operation, symbol):
        super().__init__()
        self.child = None
        self.operation = operation
        self.symbol = symbol
        self.parent = parent

    def getValue(self, variables):
        return self.operation(self.child.getValue(variables))
    
    def display(self):
        return f"{self.symbol}({self.child.display()})"