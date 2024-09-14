## README for Multi-Layer Perceptron Program

### Overview

This program is a Java-based implementation of a Multi-Layer Perceptron (MLP) designed to improve the localization accuracy of a robot inside a building using Ultra-Wideband (UWB) technology. The goal of this software is to correct inaccurate robot localization measurements using artificial neural networks. The program leverages the DeepLearning4J (DL4J) and ND4J libraries to build, train, and evaluate various MLP models.

### Features
- **Model Creation**: Create a neural network with a customizable number of hidden layers and neurons, with the ability to specify the learning rate.
- **Model Training**: Train the network using a dataset to optimize localization accuracy. Users can specify the number of epochs for training.
- **Prediction and Evaluation**: The program can generate predictions on test data and save the results to a CSV file for further analysis.
- **Performance Tracking**: Monitors model performance after each training epoch to evaluate its improvement.

### Program Structure

1. **Main Class (`Main.java`)**:
    - Handles user input and executes different actions (`create`, `train`, `predict`) based on the command line arguments.

2. **Managers**:
    - **MultiLayerPerceptronManager**: Manages the creation, training, and saving/loading of the neural network model.
    - **DataManager**: Manages loading and preprocessing of training and testing datasets.
    
3. **EpochScoreListener**: Provides a mechanism to track model performance during training by observing the network's score after each epoch.

### Prerequisites

- Java 8 or higher
- DL4J and ND4J libraries
- Properly formatted dataset files for training and testing

### How to Use

The program can be executed using the following commands:

1. **Create a Model**  
   Command:
   ```bash
   java -jar multilayer_perceptron.jar create <hidden_layers> <learning_rate> <path_to_save>
   ```
   Example:
   ```bash
   java -jar multilayer_perceptron.jar create 32,16,8 0.001 ./model.zip
   ```
   This will create a model with 3 hidden layers (32, 16, and 8 neurons, respectively) and a learning rate of 0.001. The model will be saved to `model.zip`.

2. **Train a Model**  
   Command:
   ```bash
   java -jar multilayer_perceptron.jar train <model_path> <training_data_path> <test_data_path> <epochs>
   ```
   Example:
   ```bash
   java -jar multilayer_perceptron.jar train ./model.zip ./train_data.csv ./test_data.csv 10
   ```
   This will train the model saved at `model.zip` using the training dataset from `train_data.csv`, and test it on `test_data.csv` over 10 epochs.

3. **Make Predictions**  
   Command:
   ```bash
   java -jar multilayer_perceptron.jar predict <model_path> <user_data_path> <csv_file_path>
   ```
   Example:
   ```bash
   java -jar multilayer_perceptron.jar predict ./model.zip ./user_data.csv ./predictions.csv
   ```
   This will load the model from `model.zip`, make predictions based on the input data from `user_data.csv`, and save the results in `predictions.csv`.

### Dataset Requirements

The dataset files used for training, testing, and prediction must be in CSV format and follow a structure similar to:

```
inputX,inputY,outputX,outputY
```

Where `inputX` and `inputY` represent the input features (e.g., UWB measurement data), and `outputX` and `outputY` are the actual or predicted positions of the robot.

### Results and Performance Analysis

From the experiments conducted in the research section, different network architectures were tested to improve localization accuracy. The table below summarizes the most effective configurations:

| Hidden Layers | Neurons   | Activation Function | Learning Rate | Epochs |
| --------------| --------- | ------------------- | ------------- | ------ |
| 1             | 32        | ReLU                | 0.0001        | 6      |
| 2             | 4, 8      | ReLU                | 0.0001        | 7      |
| 3             | 4, 8, 4   | ReLU                | 0.0001        | 10     |

### Future Work

Possible improvements to the program could include experimenting with other machine learning models or expanding the dataset to handle more diverse UWB signal conditions.

### Authors

- Jan Gluźniewicz (247665)  

### License

This project is open-source and distributed under the Apache License 2.0.
