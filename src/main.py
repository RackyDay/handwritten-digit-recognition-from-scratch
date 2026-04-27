import random
import math

training_images = 'training_data/train-images.idx3-ubyte'
training_labels = 'training_data/train-labels.idx1-ubyte'

def read_images(data_path):
    with open(data_path, 'rb') as f:
        f.read(16) #skip the header
        raw_data = f.read()

        images = [[pixel/255 for pixel in raw_data[i * 784: (i+1) * 784]]
                for i in range(60000)]
    
    return images

def read_labels(data_path):
    with open(data_path, 'rb') as f:
        f.read(8)
        return list(f.read())

def create_neural_network(layer_sizes):
    weights = []
    biases = []

    for i in range(1, len(layer_sizes)):
        weight_matrix = [[(random.random() - 0.5) * 0.1 for _ in range(layer_sizes[i-1])] for _ in range(layer_sizes[i])]
        #(current layer size) lists of length (previous layer size) i.e. 128 lists of 784, 64 lists of 128, 10 lists of 64
        
        bias_matrix = [0] * layer_sizes[i]
        weights.append(weight_matrix)
        biases.append(bias_matrix)
    
    return weights, biases

def forward_prop(weights, biases, image):

    activations = [image.copy()]
    pre_activations = [None]

    for l in range(len(weights)): #loops through every layer

        W = weights[l]
        b = biases[l]

        z = [0] * len(W)
        a = [0] * len(W)
        prev_a = activations[-1]

        for current_neuron in range(len(W)): #go through each neuron in current layer
            partial_sum = 0
            for previous_neuron in range(len(W[current_neuron])): #go through each neuron in previous layer
                partial_sum += (W[current_neuron][previous_neuron] * prev_a[previous_neuron])
            z[current_neuron] = partial_sum + b[current_neuron]

        if l == len(weights) -1:
            softmax(z, a)
        else:
            reLU(z, a)
        
        pre_activations.append(z)
        activations.append(a)

    return pre_activations, activations

def softmax(pre, post):
    exp_sum = 0
    max_val = max(pre)

    for i in range(len(pre)):
        exp_sum += math.exp(pre[i] - max_val) #subtract max so exp doesn't overflow
    
    for i in range(len(pre)):
        post[i] = math.exp(pre[i] - max_val)/exp_sum

    return

def reLU(pre, post):
    for i in range(len(pre)):
        post[i] = max(0, pre[i])
    return

def calculate_cost(label, activations):
    confidence = activations[-1][label]
    return -1 * math.log(confidence)

def back_prop(weights, biases, pre_activations, activations, label):

    dW, dB = create_gradient_accumulators(weights, biases)

    delta = activations[-1][:]
    delta[label] -=1

    for l in reversed(range(len(weights))):
        for j in range(len(delta)):
            dB[l][j] = delta[j]
            for i in range(len(activations[l])):
                dW[l][j][i] = delta[j] * activations[l][i]
    
        if l > 0:
            new_delta = [0] * len(activations[l])

            for i in range(len(new_delta)):
                partial_sum = 0
                for j in range(len(delta)):
                    product = weights[l][j][i] * delta[j]
                    partial_sum += product
                if pre_activations[l][i] > 0:
                    new_delta[i] = partial_sum 
                else:
                    new_delta[i] = 0
            delta = new_delta
    return dW, dB

def sum_gradients(acc_dW, acc_dB, dW, dB):
    for l in range(len(dW)):
        current_dW = dW[l]
        current_dB = dB[l]
        for b in range(len(current_dB)):
            acc_dB[l][b] += current_dB[b]
        
        for i in range(len(current_dW)):
            for j in range(len(current_dW[i])):
                acc_dW[l][i][j] += current_dW[i][j]

    return acc_dW, acc_dB

def update_gradients(weights, biases, acc_dW, acc_dB, learning_rate, batch_size):
    for l in range(len(weights)):
        for b in range(len(biases[l])):
            biases[l][b] -= learning_rate * (acc_dB[l][b]/batch_size)
        
        for i in range(len(weights[l])):
            for j in range(len(weights[l][i])):
                weights[l][i][j] -= learning_rate * (acc_dW[l][i][j]/batch_size)

def create_gradient_accumulators(weights, biases):

    dW = []
    dB = []

    for l in range(len(weights)):
        current_weights = weights[l]
        current_biases = biases[l]

        dB.append([0] * len(current_biases))
        dW.append([[0 for _ in range(len(current_weights[0]))] for _ in range(len(current_weights))])
    
    return dW, dB

def create_batches(training_data, batch_size):
    random.shuffle(training_data)
    for i in range(0, len(training_data), batch_size):
        yield training_data[i: i+batch_size]

def train(weights, biases, training_data, epochs, batch_size, learning_rate):

    for epoch in range(epochs):
        for index, batch in enumerate(create_batches(training_data, batch_size)):
            
            correct = 0
            total = 0
            total_cost = 0
            acc_dW, acc_dB = create_gradient_accumulators(weights, biases)

            for image, label in batch:
                pre_activations, activations = forward_prop(weights, biases, image)
                dW, dB = back_prop(weights, biases, pre_activations, activations, label)
                acc_dW, acc_dB = sum_gradients(acc_dW, acc_dB, dW, dB)
                total_cost += calculate_cost(label, activations)
                predicted = activations[-1].index(max(activations[-1]))
                if predicted == label:
                    correct +=1
                total +=1

            update_gradients(weights, biases, acc_dW, acc_dB, learning_rate, batch_size)
            print("epoch:", epoch + 1, "batch:", index + 1, "cost:", total_cost/batch_size, "accuracy:", correct/total)

    return None

images = read_images(training_images)
labels = read_labels(training_labels)

data = list(zip(images, labels))
random.shuffle(data)

train_data = data[:48000]
test_data = data[48000:]

layer_sizes = [784, 128, 64, 10]
weights, biases = create_neural_network(layer_sizes)
train(weights, biases, train_data, 8, 100, 0.1)

with open("weights.txt", "w") as f:
    f.write(str(weights))

with open("biases.txt", "w") as f:
    f.write(str(biases))

correct = 0
count = 0
for image, label in test_data:
    count +=1
    pre_activations, activations = forward_prop(weights, biases, image)
    cost = calculate_cost(label, activations)
    pred = activations[-1].index(max(activations[-1]))
    if pred == label:
        correct +=1
    print("Example", count, "Prediction:", pred, "Actual:", label, "Correct:", (pred == label))

print("Accuracy: ", (correct/12000) * 100)