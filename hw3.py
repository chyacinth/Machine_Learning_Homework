from scipy.io import loadmat
import numpy as np
import random
from numpy import linalg as LA
import math

def xor_kernel_3(x, y):
    assert x.shape == y.shape, "kernel arguments do not have the same shape"
    assert x.shape == (2,), "kernel arguments are not 1D vector"
    
    return (np.square(x[0]) - np.square(x[1])) * (np.square(y[0]) - np.square(y[1])) + \
                x[0] * x[1] * y[0] * y[1] + \
                    (np.square(x[0]) + np.square(x[1])) * (np.square(y[0]) + np.square(y[1]))

def radical_basis(x, y):
    assert x.shape == y.shape, "kernel arguments do not have the same shape"
    # print (LA.norm(x - y))
    return math.exp(-np.square(LA.norm(x - y))/(2 * np.square(radical_basis.sigma)))

class SVM:
    """
    Args:
        N: training sample sizes, equal to the length of W
        kernel: kernel function
    """
    def __init__(self, N, train_x, train_labels, kernel):
        assert train_x.shape[1] == train_labels.shape[1], "training sample size != label size"
        assert train_x.shape[1] == N, "training sample size != N"
        self.N = N
        self.kernel = kernel
        
        self.train_x = train_x
        self.labels = train_labels.reshape(N)        
        
        # weights
        self.W = np.zeros(N)
        self.b = 0
        
        # Gram matrix
        self.Gram = np.zeros((N, N))
        for i in range(self.N):
            for j in range(i+1):
                self.Gram[i][j] = kernel(self.train_x.T[i], self.train_x.T[j])
                self.Gram[j][i] = self.Gram[i][j]
                
        print(self.Gram)

        # intrinsic variables
        self.feature_sz = train_x.shape[0]
        self.C = 1
        self.epsilon = 10
        self.max_iter = 50
    
    """
    Args: 
        epsilon: training threshold
        C: weight range
    """
    
    def train(self, C=1, max_iter=50, epsilon=10):
        self.C = C
        self.max_iter = max_iter
        self.epsilon = 10
        
        iteration = 1
        
        while iteration <= self.max_iter:            
            # select Wi and Wj to update. Select those make |E1-E2| maximal
            max_E = 0
            i = 0
            for i in random.sample(range(self.N),k=self.N):
                Ei = self.pred_with_id(i) - self.labels[i]
                j = 0
                for j in range(self.N):
                    Ej = self.pred_with_id(j) - self.labels[j]
                    if abs(Ei - Ej) >= max_E:
                        max_E = abs(Ei - Ej)
                        break
                
                yi = self.labels[i]
                yj = self.labels[j]
                Wi_old = self.W[i]
                Wj_old = self.W[j]
                
                eta = self.Gram[i][i] + self.Gram[j][j] - 2 * self.Gram[i][j]
                if (eta <= 0 + 0.00001):
                    continue
                
                Wj_new = Wj_old + yj * (Ei - Ej) / eta
                
                L = 0
                H = 0
                
                if yi != yj:
                    L = max(0, Wj_old - Wi_old) 
                    H = min(self.C, self.C + Wj_old - Wi_old)
                else:
                    L = max(0, Wi_old + Wj_old - self.C) 
                    H = min(self.C, Wi_old + Wj_old)
                    
                Wj_new = self.clip(Wj_new, L, H)
                Wi_new = Wi_old + yi * yj * (Wj_old - Wj_new)
                self.W[j] = Wj_new
                self.W[i] = Wi_new                 
                
                bi_new = -Ei + Wi_old * yi * self.Gram[i][i] + Wj_old * yj * self.Gram[j][i] + self.b - \
                                    Wi_new * yi * self.Gram[i][i] - Wj_new * yj * self.Gram[j][i]
                
                bj_new = -Ej - yi * self.Gram[i][j] * (Wi_new - Wi_old) - \
                                yj * self.Gram[j][j] * (Wj_new - Wj_old) + self.b
                
                b_new = (bi_new + bj_new) / 2
                self.b = b_new
                
            iteration += 1
            print("{}th iteration finishes".format(iteration))
        
    def test(self, test_x, test_label):
        assert len(test_x.shape) == 2, "test_x shape not right"
        sz = test_x.shape[1]
        correct = 0        
        for i in range(sz):            
            if self.pred(test_x.T[i]) * test_label[0][i] > 0:
                correct += 1
        
        return correct / sz

    def pred_with_id(self, j):
        
        prediction = self.b
        
        prediction += (self.W * self.labels * self.Gram[j]).sum()
        
        return prediction
    
    """
    Args:
        x: input vector
    """
    def pred(self, x):
        assert x.shape == (self.feature_sz,), "input size not match"
        
        prediction = self.b
        
        for i in range(self.N):
            prediction += self.W[i] * self.labels[i] * self.kernel(self.train_x.T[i], x)
        
        return prediction
    
    def clip(self, W_new, L, H):
        if W_new > H:
            return H
        if W_new < L:
            return L
        return W_new

def generate_data(category, train_x, train_labels, test_x, test_labels, 
                    pos_size = None, false_size = None, test_size = None):
    # Generating training data for SVM

    temp_train_x_pos = train_x[:, (train_labels == category)[0]]
    temp_train_label_pos = train_labels[:, (train_labels == category)[0]]
    temp_train_label_pos[np.nonzero(temp_train_label_pos)] = 1
    
    if pos_size == None:
        pos_size = temp_train_x_pos.shape[1]
    selected_rows = random.sample(range(pos_size),k=pos_size)

    temp_train_x_pos = temp_train_x_pos[:, selected_rows]
    temp_train_label_pos = temp_train_label_pos[:, selected_rows]

    sample_size_pos = temp_train_x_pos.shape[1]

    temp_train_x_false = train_x[:, (train_labels != category)[0]]
    temp_train_label_false = train_labels[:, (train_labels != category)[0]]
    temp_train_label_false[np.nonzero(temp_train_label_false)] = -1

    if false_size == None:
        false_size = temp_train_x_false.shape[1]        
    selected_rows = random.sample(range(false_size),k=false_size)

    #sample_size_false = temp_train_label_false.shape[1]

    #selected_rows = random.sample(range(sample_size_false),k=3 * sample_size_pos)

    temp_train_x_false = temp_train_x_false[:, selected_rows]
    temp_train_label_false = temp_train_label_false[:, selected_rows]

    svm_train_x = np.concatenate((temp_train_x_pos, temp_train_x_false), axis = 1)
    svm_train_label = np.concatenate((temp_train_label_pos, temp_train_label_false), axis = 1)
    
    
    # Generating testing data for SVM    
    temp_test_x_pos = test_x[:, (test_labels == category)[0]]
    temp_test_label_pos = test_labels[:, (test_labels == category)[0]]
    temp_test_label_pos[np.nonzero(temp_test_label_pos)] = 1

    sample_size_pos = temp_test_x_pos.shape[1]

    temp_test_x_false = test_x[:, (test_labels != category)[0]]
    temp_test_label_false = test_labels[:, (test_labels != category)[0]]
    temp_test_label_false[np.nonzero(temp_test_label_false)] = -1

    #sample_size_false = temp_test_label_false.shape[1]

    #selected_rows = random.sample(range(sample_size_false),k=sample_size_pos)

    #temp_test_x_false = temp_test_x_false[:, selected_rows]
    #temp_test_label_false = temp_test_label_false[:, selected_rows]

    svm_test_x = np.concatenate((temp_test_x_pos, temp_test_x_false), axis = 1)
    svm_test_label = np.concatenate((temp_test_label_pos, temp_test_label_false), axis = 1)
    
    final_train_sz = svm_train_x.shape[1]

    selected_rows = random.sample(range(final_train_sz),k=final_train_sz)
    svm_train_x = svm_train_x.T[selected_rows].T
    svm_train_label = svm_train_label.T[selected_rows].T

    if test_size == None:
        test_size = svm_test_x.shape[1]

    selected_rows = random.sample(range(test_size),k=test_size)

    svm_test_x = svm_test_x[:, selected_rows]
    svm_test_label = svm_test_label[:, selected_rows]

    return (svm_train_x, svm_train_label, svm_test_x, svm_test_label)

def test_xor():
    train_x = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1]])
    train_labels = np.array([[-1, 1, 1, -1]])    
    sample_nums = 4
    
    svm = SVM(sample_nums, train_x, train_labels, xor_kernel_3)
    
    svm.train(C=1, max_iter=30)
    print(svm.test(train_x, train_labels))

def test_mnist():
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

    svm_train_xs = []
    svm_train_labels = []
    svm_test_xs = []
    svm_test_labels = []

    category = 1
    positive_sample_num = 1000
    false_sample_num = 3000
    test_num = None
    
    print("Start generating data...")
    (svm_train_x, svm_train_label, svm_test_x, svm_test_label) = \
        generate_data(category, train_x, train_labels, test_x, test_labels, 
        positive_sample_num, false_sample_num, test_num)
    print("Generating data successful...")
    print(svm_train_x.shape)
    print(svm_test_x.shape)

    sample_nums = svm_train_x.shape[1]

    radical_basis.sigma = 8
    print("Init SVM and GRAM Matrix...")
    svm = SVM(sample_nums, svm_train_x, svm_train_label, radical_basis)
    print("Initialization successful")
    print("Start training...")
    svm.train(max_iter=100)
    print("Training successful")

    #print(svm.test(svm_train_x, svm_train_label))
    print("Start testing...")
    print(svm.test(svm_test_x, svm_test_label))
    print("Testing successful")

test_mnist()