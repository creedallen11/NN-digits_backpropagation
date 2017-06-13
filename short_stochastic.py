#python3

import numpy as np

#np.random.seed(1)

def importDigits(file):
	"""Import digits from CSV, map labels to list of 10 elements
	set to 0, set true label to 1. Return Train, Label. """
	f = open(file)
	#n = 9
	feature_train = []
	label_train = []
	for line in f:
		feature, label = line.split(",")
		feature_train.append([int(c) for c in feature.split(" ")])
		l = [0] * 10
		l[int(label)] = 1
		label_train.append(l)
	return np.array(feature_train), np.array(label_train)


# def listToLabel(lst):
# 	"""Converts the 10 dim output of NN to a single label"""
# 	return lst.index(max(lst))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

# alternative to sigmoid activation, sometimes works better
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - np.tanh(x)**2

class NN:
    def __init__(self, nI=256, nH=30, nO=10):
        # Randomly initialize our network weights with mean 0
        # This implicitly initializes nodes, which become explicit in runNN
        # first Input layer (nI) has 3 inputs, second Hidden (nH) has 4 neurons
        self.syn0 = 2*np.random.random((nI, nH))-1
        # from the 4 Hidden to the 1 Output (nO)
        self.syn1 = 2*np.random.random((nH, nO))-1

    def runNN(self, X):
        # Feed forward activity through layers 0, 1, and 2
        self.l0 = X
        self.l1 = sigmoid(np.dot(self.l0, self.syn0))
        self.l2 = sigmoid(np.dot(self.l1, self.syn1))

        return self.l2

    def backPropagate(self, y, N):
        # by how much did we miss the target value?
        # could use a variety of errors here, e.g., MSE, etc
        l2_error = y - self.l2
        # print(sum(l2_error))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * sigmoid_prime(self.l2)

        # how much did each l1 value contribute to the l2 error,
        # according to the weights?
        l1_error = l2_delta.dot(self.syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * sigmoid_prime(self.l1)

        # update weights with learning rate = N
        self.syn1 += (self.l1.T.dot(l2_delta)) * N
        self.syn0 += (self.l0.T.dot(l1_delta)) * N

    def train(self, X, y, max_iterations=100, N=.75):
    	features_split = np.split(X, X.shape[0])
    	labels_split = np.split(y, y.shape[0])
    	
    	#train online
    	for f, l in zip(features_split, labels_split):
    	   for round in range(max_iterations):
                self.runNN(f)
                self.backPropagate(l, N)

    def test(self, X, y):
        final_prediction = self.runNN(X)
        print('Final classification outputs are: ')
        print(final_prediction)
        print('Final classification outputs was supposed to be: ')
        print(y)
        print('mean error is: ', np.mean((final_prediction - y)**2))
        return np.mean((final_prediction - y)**2)

def main():
    # example trial run of 
    print("Generating training data...")
    feature_train, label_train = importDigits("digits.csv")
    print("Creating Neural Network...")
    myNN = NN()
    print ("Training Neural Network on data...")
    myNN.train(feature_train, label_train)
    print("Testing Neural Network...")
    myNN.test(feature_train, label_train)
    print("Done.")


if __name__ == "__main__":
    main()