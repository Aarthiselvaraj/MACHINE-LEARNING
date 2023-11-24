import numpy as np
X = np.array(([2,9],[1,5],[3,6]), dtype = float)
Y = np.array(([92],[86],[89]), dtype = float)

X = X/np.amax(X,axis = 0)
Y = Y/100

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_grad(x):
    return x*(1-x)
epoch = 1000

input_neurons = 2
hidden_neurons = 3
output_neurons = 1
w = np.random.uniform(size=(input_neurons,hidden_neurons))
b= np.random.uniform(size=(1,hidden_neurons))

wout = np.random.uniform(size = (hidden_neurons,output_neurons))
bout = np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    h_act = sigmoid(X.dot(w) + b)
    output = sigmoid(h_act.dot(wout) + bout)
    
print("Normalized Input : \n", str(X))
print("Actual Output : \n", str(Y))
print("Predicted Output \n: ",output)
