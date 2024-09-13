package me.jangluzniewicz.multilayer_perceptron.managers;

import me.jangluzniewicz.multilayer_perceptron.logic.MultiLayerPerceptron;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;

public class MultiLayerPerceptronManager {
    private final MultiLayerPerceptron multiLayerPerceptron;
    public MultiLayerPerceptronManager(int[] hiddenLayers, double learningRate) {
        multiLayerPerceptron = new MultiLayerPerceptron(hiddenLayers, learningRate);
    }

    public MultiLayerNetwork createModel() {
        return multiLayerPerceptron.createModel();
    }

    public static void trainModel(MultiLayerNetwork model, DataSet trainData, int batchSize) {
        MultiLayerPerceptron.trainModel(model, trainData, batchSize);
    }

    public static void saveModel(MultiLayerNetwork model, String filePath) throws IOException {
        ModelSerializer.writeModel(model, new File(filePath), true);
    }

    public static MultiLayerNetwork loadModel(String filePath) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
    }
}
