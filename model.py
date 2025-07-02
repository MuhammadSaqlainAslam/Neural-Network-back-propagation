
from matplotlib import pyplot as plt
import numpy as np

class NeuralNetwork:

    def __init__(self, hidden_size):
        self.input_size = 2
        self.hidden_size = hidden_size
        self.output_size = 1

        self.weights_hidden_layer1 = np.random.randn(self.input_size, self.hidden_size) # 2*10
        # self.weights_hidden_layer2 = np.random.randn(self.hidden_size, self.hidden_size) # 10*2
        self.weights_hidden_layer2 = np.random.randn(self.hidden_size, 2) # 10*2
        # self.weights_hidden_layer3 = np.random.randn(self.hidden_size, self.hidden_size)  
        self.weights_hidden_layer3 = np.random.randn(2, 3) # 2*3
        self.weights_hidden_layer4 = np.random.randn(3, 10)  # 3*10
        #self.weights_hidden_layer4 = np.random.randn(self.hidden_size, self.hidden_size)
        self.weights_hidden_layer5 = np.random.randn(self.hidden_size, self.hidden_size)
        self.weights_hidden_layer6 = np.random.randn(self.hidden_size, self.hidden_size)
        self.weights_hidden_layer7 = np.random.randn(self.hidden_size, self.output_size)

      
        self.hidden1_bias = np.zeros((1, self.hidden_size)) # 1*10
        self.hidden2_bias = np.zeros((1, 2)) 
        self.hidden3_bias = np.zeros((1, 3))
        self.hidden4_bias = np.zeros((1, self.hidden_size))
        self.hidden5_bias = np.zeros((1, self.hidden_size))
        self.hidden6_bias = np.zeros((1, self.hidden_size))
        self.hidden7_bias = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return np.multiply(x, 1.0 - x)
    
    #def relu(self, x):
       # return np.maximum(0, x)

    #def derivative_relu(self, x):
    #   return np.where(x > 0, 1, 0)
    
    def feedforward(self, X):
        self.hidden_activation1 = np.dot(X, self.weights_hidden_layer1) + self.hidden1_bias
        self.hidden_output1 = self.sigmoid(self.hidden_activation1)
        #self.hidden_output1 = self.relu(self.hidden_activation1)
        #self.hidden_output1 = self.hidden_activation1 #without activation function
        #print("hidden_output1:", self.hidden_output1.shape)

        self.output_activation2 = np.dot(self.hidden_output1, self.weights_hidden_layer2) + self.hidden2_bias
        self.hidden_output2 = self.sigmoid(self.output_activation2)
        #self.hidden_output2 = self.relu(self.output_activation2)
        #self.hidden_output2 = self.output_activation2
        #print("hidden_output2:", self.hidden_output2.shape)

        self.output_activation3 = np.dot(self.hidden_output2, self.weights_hidden_layer3) + self.hidden3_bias
        self.hidden_output3 = self.sigmoid(self.output_activation3)
        #self.hidden_output3 = self.relu(self.output_activation3)
        #self.hidden_output3 = self.output_activation3
        #print("hidden_output3:", self.hidden_output3.shape)

        self.output_activation4 = np.dot(self.hidden_output3, self.weights_hidden_layer4) + self.hidden4_bias
        self.hidden_output4 = self.sigmoid(self.output_activation4)
        #self.hidden_output4 = self.relu(self.output_activation4)
        #self.hidden_output4 = self.output_activation4
        #print("hidden_output4:", self.hidden_output4.shape)

        self.output_activation5 = np.dot(self.hidden_output4, self.weights_hidden_layer5) + self.hidden5_bias
        self.hidden_output5= self.sigmoid(self.output_activation5)

        self.output_activation6 = np.dot(self.hidden_output5, self.weights_hidden_layer6) + self.hidden6_bias
        self.hidden_output6= self.sigmoid(self.output_activation6)


        self.output_activation7 = np.dot(self.hidden_output6, self.weights_hidden_layer7) + self.hidden7_bias
        predicted_output = self.sigmoid(self.output_activation7)
        #predicted_output = self.relu(self.output_activation5)
        #predicted_output = self.output_activation5  #without activation function
        #print("Output:", predicted_output.shape)

        return predicted_output
    
    def backward(self, X, y, predicted_output, learning_rate=1e-4):
        # Output layer error and delta
        output_error = y - predicted_output
        output_delta = output_error * self.derivative_sigmoid(predicted_output)
        #output_delta = output_error * self.derivative_relu(predicted_output)
        
        hidden_error6 = np.dot(output_delta, self.weights_hidden_layer7.T)
        hidden_delta6 = hidden_error6 * self.derivative_sigmoid(self.hidden_output6)

        hidden_error5 = np.dot(hidden_delta6, self.weights_hidden_layer6.T)
        hidden_delta5 = hidden_error5 * self.derivative_sigmoid(self.hidden_output5)
        
        # 4th hidden layer (hidden_output3 -> output)
        hidden_error4 = np.dot(hidden_delta5, self.weights_hidden_layer5.T)
        hidden_delta4 = hidden_error4 * self.derivative_sigmoid(self.hidden_output4)

        # 3rd hidden layer (hidden_output3 -> output)
        hidden_error3 = np.dot(hidden_delta4, self.weights_hidden_layer4.T)
        hidden_delta3 = hidden_error3 * self.derivative_sigmoid(self.hidden_output3)
        #hidden_delta3 = hidden_error3 * self.derivative_relu(self.hidden_output3)

        # 2nd hidden layer (hidden_output2 -> output)
        hidden_error2 = np.dot(hidden_delta3, self.weights_hidden_layer3.T)
        hidden_delta2 = hidden_error2 * self.derivative_sigmoid(self.hidden_output2)
        #hidden_delta2 = hidden_error2 * self.derivative_relu(self.hidden_output2)

        # 1stnd hidden layer (hidden_output1 -> hidden_output2)
        hidden_error1 = np.dot(hidden_delta2, self.weights_hidden_layer2.T)
        hidden_delta1 = hidden_error1* self.derivative_sigmoid(self.hidden_output1)
        #hidden_delta1 = hidden_error1* self.derivative_relu(self.hidden_output1)

        self.weights_hidden_layer7 += np.dot(self.hidden_output6.T, output_delta) * learning_rate
        self.hidden7_bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_layer6 += np.dot(self.hidden_output5.T, hidden_delta6) * learning_rate
        self.hidden6_bias += np.sum(hidden_delta6, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_layer5 += np.dot(self.hidden_output4.T, hidden_delta5) * learning_rate
        self.hidden5_bias += np.sum(hidden_delta5, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_layer4 += np.dot(self.hidden_output3.T, hidden_delta4) * learning_rate
        self.hidden4_bias += np.sum(hidden_delta4, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_layer3 += np.dot(self.hidden_output2.T, hidden_delta3) * learning_rate
        self.hidden3_bias += np.sum(hidden_delta3, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_layer2 += np.dot(self.hidden_output1.T, hidden_delta2) * learning_rate
        # self.weights_hidden_layer2 += np.dot(self.weights_hidden_layer2, output_delta) * learning_rate
        self.hidden2_bias += np.sum(hidden_delta2, axis=0, keepdims=True) * learning_rate
        
        self.weights_hidden_layer1 += np.dot(X.T, hidden_delta1) * learning_rate
        # self.weights_hidden_layer1 += np.dot(X.T, self.weights_hidden_layer1) * learning_rate
        self.hidden1_bias += np.sum(hidden_delta1, axis=0, keepdims=True) * learning_rate
   
    def train(self, X, y, epochs=10000, learning_rate=1e-2):
        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, output, learning_rate)
            # self.backward(X, y, learning_rate)
            #if epoch % 1000 == 0:
                #loss = np.mean(np.square(y - output))
                #print(f"Epoch {epoch}, Loss: {loss:f}")
            loss = np.mean(np.square(y - output))
            
            self.loss_history.append(loss)
            predicted_classes = np.round(output)
            accuracy = np.mean(predicted_classes == y)
            self.accuracy_history.append(accuracy)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.7f}, Accuracy: {accuracy:.7f}")

            #predicted_classes = np.round(output)
            #accuracy = np.mean(predicted_classes == y)
            #print(f"Epoch {epoch + 1:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.7f}")
    
    def evaluate_accuracy(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        predictions = self.feedforward(X)
        predicted_classes = np.round(predictions)
        accuracy = np.mean(predicted_classes == y) * 100
        return accuracy

    
    
    @staticmethod
    def show_result(x, y, pred_y):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title ('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i]==0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        plt.show()
    

def binarized (pred):
    #breakpoint()
    lst= []
    for idx in range(len(pred)):
        if pred[idx] >= 0.5:
            lst.append(1)
        else:
            lst.append(0)
    return lst



if __name__ == "__main__":
    from Dataset import generate_linear, generate_XOR_easy
    #x, y = generate_linear(n=100)
    x, y = generate_XOR_easy()

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    nn = NeuralNetwork(hidden_size= 10)
    print("Training the network...")
    nn.train(x, y, epochs=10000, learning_rate=0.1)


    pred = nn.feedforward(x)
    
    binary_pred = binarized(pred)

    accuracy = nn.evaluate_accuracy(x, y)
    print(f"Accuracy: {accuracy:.7f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(nn.loss_history, label='Loss')
    plt.plot(nn.accuracy_history, label='Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Loss & Accuracy vs Epoch)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    NeuralNetwork.show_result(x, y, binary_pred)


    #nn.backward(x[0], y[0], pred)
    #pred = nn.feedforward(x[0])
    #print(f"after bp loss val : {y[0] - pred}")
    

    