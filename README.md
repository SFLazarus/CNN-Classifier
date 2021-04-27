# CNN-Classifier
CNN Classifier written from scratch in python without using any libraries like Keras and Tensorflow

# Says One Neuron To Another
## Neural network architectures
1. Set up a new git repository in your GitHub account
2. Pick two datasets from
https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
3. Choose a programming language (Python, C/C++, Java)
4. Formulate ideas on how neural networks can be used to accomplish the task for the specific dataset
5. Build a neural network to model the prediction process programmatically
6. Document your process and results
7. Commit your source code, documentation and other supporting files to the git repository in GitHub

---

# Dataset 1:

`tf.keras.datasets.mnist.load_data(path="mnist.npz")`

- This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. 
- x_train, x_test: uint8 arrays of grayscale image data with shapes (num_samples, 28, 28).
- y_train, y_test: uint8 arrays of digit labels (integers in range 0-9) with shapes (num_samples,).
- License: Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

- The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

- Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

# Dataset 2:

`tf.keras.datasets.fashion_mnist.load_data()`

- This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST. The class labels are:

- Label	Description

- 0	T-shirt/top
- 1	Trouser
- 2	Pullover
- 3	Dress
- 4	Coat
- 5	Sandal
- 6	Shirt
- 7	Sneaker
- 8	Bag
- 9	Ankle boot

- Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

- License: The copyright for Fashion-MNIST is held by Zalando SE. Fashion-MNIST is licensed under the MIT license.

- Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

---

### Structure of Neural Network

- Input Layer has 784 neurons(28 x 28)
- Hidden Layer has 15 neurons
- Output Layer has 10 neurons(10 classes)
- Activation Function used: sigmoid function
- Softmax function is used in Output layer
- Loss Function: Sum of Squares.

---

# Performance of Fashion Classifier:

## After Training we had an global maxima accuracy of 80%
![image](https://github.com/SFLazarus/CNN-Classifier/blob/main/reports/fashion_calssifier_train_acc_loss_plot.png)


## After Testing our model we have an accuracy of 70.77%
- Here are some of the predictions:
![image](https://github.com/SFLazarus/CNN-Classifier/blob/main/reports/fashion_classifier.png)

---

# Performance of Diit Recognizer:

## After Training we had an global maxima accuracy of 97%
![image](https://github.com/SFLazarus/CNN-Classifier/blob/main/reports/digit_recognizer_train_acc_loss_plot.png)


## After Testing our model we have an accuracy of 77.79%
- Here are some of the predictions:
![image](https://github.com/SFLazarus/CNN-Classifier/blob/main/reports/number_predictions.png)


# Results and Future Developments:

- Our model trained pretty well but we could not obtain similar train and test accuracies this might be because of overfitting or not enough data to train our model or a shallow network with only one hidden layer.
- In future we can try to implement a deeper network with more hidden layers and also instead of initializing weights and biases with random values we can use some standard weights which are trained on more powerful machines.


# Project Structure:
### Readme.md
- Project description
### Notebooks
- Jupyter Notebook implementing Digit-Recognizer model.
- Jupyter Notebook implementing FashionClassifier model.
### Reports
- Visualizations in png format images 
### Requirements.txt
- Info about Tools, frameworks and libraries required to reproduce the work flow
