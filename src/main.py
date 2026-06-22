import random
import math
import numpy as np
import matplotlib.pyplot as plt

training_images = 'training_data/train-images.idx3-ubyte'
training_labels = 'training_data/train-labels.idx1-ubyte'

def read_images(data_path):
    with open(data_path, 'rb') as f:
        f.read(16) #skip the header
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = raw_data.reshape(-1, 784).astype(np.float32)/255
    
    return images

def read_labels(data_path):
    with open(data_path, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

def create_neural_network(layer_sizes):

    weights = []
    biases = []

    for i in range(1, len(layer_sizes)):

        weight_matrix = np.random.uniform(-0.05, 0.05, (layer_sizes[i], layer_sizes[i-1])) #matrix with # of neurons in current layer rows and # of neurons in previous layer columns
        weights.append(weight_matrix)

        bias_matrix = np.zeros(layer_sizes[i])
        biases.append(bias_matrix)
    
    return weights, biases

def forward_prop(weights, biases, image):

    activations = [image.copy()]
    pre_activations = [None]

    for l in range(len(weights)): #loops through every layer

        W = weights[l]
        b = biases[l]
        
        prev_a = activations[-1] #initially image
        
        z = W @ prev_a + b

        if l == len(weights) -1: #final layer
            a = softmax(z)
        else:
            a = reLU(z)
        
        pre_activations.append(z)
        activations.append(a)

    return pre_activations, activations

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z/np.sum(exp_z)

def reLU(z):
    return np.maximum(0, z)

def calculate_cost(label, activations):
    confidence = activations[-1][label]
    return -np.log(confidence)

def back_prop(weights, biases, pre_activations, activations, label):

    dW, dB = create_gradient_accumulators(weights, biases)

    delta = activations[-1].copy()
    delta[label] -=1 #derivative of the loss with respect to the output layer pre activations

    for l in reversed(range(len(weights))):
        dB[l] = delta
        dW[l] = np.outer(delta, activations[l])
    
        if l > 0:
            new_delta = weights[l].T @ delta
            delta = new_delta * (pre_activations[l] > 0)

    return dW, dB

def sum_gradients(acc_dW, acc_dB, dW, dB):

    for l in range(len(dW)):
        acc_dW[l] += dW[l]
        acc_dB[l] += dB[l]
    
    return acc_dW, acc_dB

def update_gradients(weights, biases, acc_dW, acc_dB, learning_rate, batch_size):

    for l in range(len(weights)):
        biases[l] -= learning_rate * (acc_dB[l]/batch_size)
        weights[l] -= learning_rate * (acc_dW[l]/batch_size)

def create_gradient_accumulators(weights, biases):

    dW = [np.zeros_like(W) for W in weights]
    dB = [np.zeros_like(b) for b in biases]
    
    return dW, dB

def create_batches(training_data, batch_size):
    random.shuffle(training_data)
    for i in range(0, len(training_data), batch_size):
        yield training_data[i: i+batch_size]

def train(weights, biases, images, labels, epochs, batch_size, learning_rate):

    n = len(labels)

    for epoch in range(epochs):
        
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            correct = 0
            batch_loss = 0

            acc_dW, acc_dB = create_gradient_accumulators(weights, biases)

            for i in batch_idx:
                image = images[i]
                label = labels[i]

                pre_activations, activations = forward_prop(weights, biases, image)
                dW, dB = back_prop(weights, biases, pre_activations, activations, label)
                acc_dW, acc_dB = sum_gradients(acc_dW, acc_dB, dW, dB)

                if np.argmax(activations[-1]) == label:
                    correct += 1
                
                batch_loss += calculate_cost(label, activations)

            update_gradients(weights, biases, acc_dW, acc_dB, learning_rate, batch_size)
            acc = correct / len(batch_idx)
            loss = batch_loss / len(batch_idx)
            print(f"epoch: {epoch + 1} batch: {start//batch_size} cost: {loss:.4f} accuracy: {acc:.2f}")

    return None

def test(test_images, test_labels, weights, biases):
    counter = 0
    for i in range(len(test_images)):
        test_image = test_images[i]
        test_label = test_labels[i]

        pre_activations, activations = forward_prop(weights, biases, test_image)

        if np.argmax(activations[-1]) == test_label:
            counter +=1
        
    return (counter/12000) * 100

def show_image(image, label, prediction):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {label}, Pred: {prediction}")
    plt.axis('off')
    plt.show()

def save_model(weights, biases):
    model = {}

    for i, (W, b) in enumerate(zip(weights, biases), start = 1):
        model[f"W{i}"] = W
        model[f"b{i}"] = b
    
    np.savez("model.npz", **model)

def load_model(model):

    data = np.load(model, allow_pickle=True)
    weights = []
    biases = []

    i = 1
    while f"W{i}" in data:
        weights.append(data[f"W{i}"])
        biases.append(data[f"b{i}"])
        i +=1
    
    return weights, biases

def predict(image, weights, biases):
    a = image

    for i in range(len(weights) - 1):
        W = weights[i]
        b = biases[i]
        a = reLU(W @ a + b)
    
    return softmax(weights[-1] @ a + biases[-1])

'''
images = read_images(training_images)
labels = read_labels(training_labels)

train_images = images[:48000]
train_labels = labels[:48000]

test_images = images[48000:]
test_labels = labels[48000:]

layer_sizes = [784, 128, 64, 10]
weights, biases = create_neural_network(layer_sizes)

train(weights, biases, train_images, train_labels, 8, 100, 0.1)

accuracy = test(test_images, test_labels, weights, biases)
print(f"Accuracy on test data: {accuracy}%")

save_model(weights, biases)
'''
images = read_images(training_images)
labels = read_labels(training_labels)

train_images = images[:48000]
train_labels = labels[:48000]

test_images = images[48000:]
test_labels = labels[48000:]

weights, biases = load_model("model.npz")

for i in range(len(test_images)):
    output = predict(test_images[i], weights, biases)

    print(f"Predicted: {np.argmax(output)}, Expected: {test_labels[i]}") if not np.argmax(output) == test_labels[i] else None

