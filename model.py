import pandas as pd
import numpy as np
from NeuralNetwork.NeuralNetwork import NeuralNetwork
import random
import matplotlib.pyplot as plt


class Iris:

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def extract(self):
        name = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']
        iris_df = pd.read_csv("iris.data", names = name)
        ip = iris_df.to_numpy()

        for i in range(len(ip)):
            self.inputs.append([])
            self.outputs.append([])
            for j in range(4):
                self.inputs[i].append(ip[i][j])

            if(ip[i][4] == 'Iris-setosa'):
                self.outputs[i].append([1,0,0])

            elif(ip[i][4] == 'Iris-versicolor'):
                self.outputs[i].append([0,1,0])

            elif(ip[i][4] == 'Iris-virginica'):
                self.outputs[i].append([0,0,1])

    def visualize(self):
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Petal Length (cm)")
        for i in range(149):
            if(self.outputs[i] == [[1, 0, 0]]):
                plt.scatter(self.inputs[i][0], self.inputs[i][2], color = "Red")
            elif(self.outputs[i] == [[0, 1, 0]]):
                plt.scatter(self.inputs[i][0], self.inputs[i][2], color = "Blue")
            elif(self.outputs[i] == [[0, 0, 1]]):
                plt.scatter(self.inputs[i][0], self.inputs[i][2], color = "Green")
        plt.show()
        print("Continue ?")
        choice = int(input())

        if(choice == 1):
            plt.xlabel("Sepal Width (cm)")
            plt.ylabel("Petal Width (cm)")

            for i in range(149):
                print(i)
                if(self.outputs[i] == [[1, 0, 0]]):
                    plt.scatter(self.inputs[i][1], self.inputs[i][3], color = "Red")
                elif(self.outputs[i] == [[0, 1, 0]]):
                    plt.scatter(self.inputs[i][1], self.inputs[i][3], color = "Blue")
                elif(self.outputs[i] == [[0, 0, 1]]):
                    plt.scatter(self.inputs[i][1], self.inputs[i][3], color = "Green")
            plt.show()



    def train(self, lr, iterations):
        global nn
        for i in range(iterations):
            print("Iteration : " + str(i))
            ip = random.randrange(0, 149)
            nn.train(self.inputs[ip], self.outputs[ip], lr)

        print("NN Trained for " + str(iterations) + " iterations")

    def test(self):
        global nn
        while(1):
            ip = random.randrange(0, 149)
            print("Test Input : ")
            print(self.inputs[ip])
            print("\nTest Output : ")
            print(self.outputs[ip])

            if(self.outputs[ip] == [[1, 0, 0]]):
                print("Iris-sentosa")
            elif(self.outputs[ip] == [[0, 1, 0]]):
                print("Iris-versicolor")
            elif(self.outputs[ip] == [[0, 0, 1]]):
                print("Iris-virginica")

            print("\nPredicted Output : ")
            op = nn.feedforward(self.inputs[ip])
            if(op.index(max(op)) == 0):
                print("Iris-sentosa")
            elif(op.index(max(op)) == 1):
                print("Iris-versicolor")
            elif(op.index(max(op)) == 2):
                print("Iris-virginica")

            print("Continue ? (1 - Y 2 - N)")
            choice = int(input())

            if(choice == 2):
                break;

iris = Iris()
iris.extract()
nn = NeuralNetwork(4, 10, 3)
# iris.visualize()
print("Training Model")
iris.train(0.1, 30000)
iris.test()
