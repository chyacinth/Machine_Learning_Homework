import numpy as np
import random
from scipy.io import loadmat
import argparse
import cProfile

def sigmoid(x, deri=False):
    if deri:
        fx = sigmoid(x)
        return fx * (1 - fx)
    else:
        return 1/(1 + np.exp(-x))

class Layer:
    def __init__(self, in_num, out_num, activation, lr, bias):
        self.bias = bias
        if not self.bias:
            self.in_num = in_num
            self.out_num = out_num
            self.lr = lr
            self.activation = activation
            self.W = np.random.uniform(-1, 1, (self.out_num, self.in_num))
            self.input = None
            self.output = None    
            self.derivative = np.zeros(self.out_num, self.in_num)
            self.gpwx = None
        else:
            self.in_num = in_num + 1
            self.out_num = out_num
            self.lr = lr
            self.activation = activation            
            self.W = np.random.uniform(-1, 1, (self.out_num, self.in_num))            
            # print(self.W)
            self.input = None
            self.output = None
            self.derivative = np.zeros((self.out_num, self.in_num))
            self.gpwx = None            

        self.wx = None

    def forward(self, inp):
        if self.bias:
            inp = np.append(inp, 1)

        self.input = inp
        #self.input = np.append(self.input, 1)
        self.wx = np.matmul(self.W, self.input)
        self.output = self.activation(self.wx)
        return self.output
    
    def update_delta_w(self, lam):        
        assert not self.input is None, "Input is None!"        
        #wx = np.matmul(self.W, self.input)
        self.gpwx = self.activation(self.wx, deri=True)
        gpld = self.gpwx * lam

        self.derivative = np.multiply(gpld[:, None], np.tile(self.input, (self.out_num, 1)))

        #if self.bias:
            #reduced_inp = self.input[:-1]
            #self.derivative[:, :-1] = np.multiply(gpld[:, None], np.tile(reduced_inp, (self.out_num, 1)))
            #self.derivative[:, -1] = gpld
            
        #else:
        #    self.derivative = np.multiply(gpld[:, None], np.tile(self.input, (self.out_num, 1)))
    
    def get_lambda(self, lam):        
        if not self.bias:
            return np.matmul(-self.W.T, self.gpwx)
        else:
            return np.matmul(-self.W.T, self.gpwx)[:-1]

    
    def update_w(self):
        self.W = self.W - self.lr * self.derivative
        #self.derivative(0)

class Network:
    def __init__(self, neurals, activation, label_nums, lr=0.05, bias=False):
        self.bias = bias
        self.label_nums = int(label_nums)
        self.layers = []
        for inp, outp in zip(neurals, neurals[1:]):
            self.layers.append(Layer(inp, outp,activation, lr, self.bias))

    def train_once(self, inp, exp, one_hot = False):
        if one_hot:
            exp_hot = np.zeros(self.label_nums)
            exp_hot[exp] = 1
            exp = exp_hot

        output = self.forward(inp)
        self.back_propagation(output, exp)
        return ((output - exp)**2).mean()

    def forward(self, inp, one_hot = False):        
        output = inp
        for layer in self.layers:
            output = layer.forward(output)
        if one_hot:
            return np.argmax(output)
        return output

    def back_propagation(self, output, expect):
        lam = output - expect
        for layer in reversed(self.layers):
            layer.update_delta_w(lam)
            lam = layer.get_lambda(lam)

        for layer in self.layers:
            layer.update_w()

def prepare_test_data(train_size, test_size):    
    x = loadmat('digits.mat')
    test_imgs = x['testImages']
    train_imgs = x['trainImages']
    test_labels = x['testLabels'].astype(np.int8)
    train_labels = x['trainLabels'].astype(np.int8)    

    test_imgs = test_imgs.reshape(784, 10000)
    train_imgs = train_imgs.reshape(784, 60000)
    avg = train_imgs.mean(axis=1).reshape(784)
    maxi = train_imgs.max(axis=1).reshape(784)
    mini = train_imgs.min(axis=1).reshape(784)
    maxmin = (maxi[:, None] - mini[:, None])
    test_x = np.divide((test_imgs - avg[:, None]), maxmin, where=maxmin!=0)
    train_x = np.divide((train_imgs - avg[:, None]), maxmin, where=maxmin!=0)
    
    test_size_all = test_x.shape[1]
    train_size_all = train_x.shape[1]

    test_selected_rows = random.sample(range(test_size_all),k=test_size)
    train_selected_rows = random.sample(range(train_size_all),k=train_size)

    return train_x[:, train_selected_rows], train_labels[:, train_selected_rows], \
        test_x[:, test_selected_rows], test_labels[:, test_selected_rows]

def test(network, test_x, test_labels, test_size):    
    sz = test_size
    correct = 0
    for i in range(sz):
        prediction = network.forward(test_x.T[i], one_hot=True)
        label = test_labels[0][i]        
        if prediction == label:
            correct += 1
        else:
            pass
            #print(prediction)
            #print(label)
            #print()
    print("plain accuracy: {}".format(correct / sz))

def main():
    #random.seed(3614)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-network",
        nargs="+",  # expects ≥ 0 arguments
        type=int,
        # default=[35, 40, 50, 60, 70, 80, 90],  # default list if no arg value
    )
    parser.add_argument(
        "-train_size",
        nargs="?",  # expects ≥ 0 arguments
        type=int,
    )
    parser.add_argument(
        "-test_size",
        nargs="?",  # expects ≥ 0 arguments
        type=int,
    )
    parser.add_argument(
        "-lr",
        nargs="?",  # expects ≥ 0 arguments
        type=float,
    )
    parser.add_argument(
        "-epochs",
        nargs="?",  # expects ≥ 0 arguments
        type=int,
    )    

    args = parser.parse_args()
    train_size = args.train_size
    test_size = args.test_size
    structure = args.network
    lr = args.lr
    epochs = args.epochs
    print(structure)

    train_x, train_labels, test_x, test_labels = prepare_test_data(train_size, test_size)
    network = Network(structure, sigmoid, 10, lr, bias=True)
    for epoch in range(epochs):
        indexes = random.sample(range(train_size),k=train_size)
        for i in indexes:
            mse = network.train_once(train_x.T[i],train_labels[0][i],one_hot=True)
                    
        if ((epoch + 1) % 1 ==0):
            print("epoch {} completes".format(epoch + 1))
            print("mse is: {}".format(mse))
    
    #test(network, test_x, test_labels, test_size)
    test(network, train_x, train_labels, train_size)

if __name__ == "__main__":
    #cProfile.run('main()')
    main()