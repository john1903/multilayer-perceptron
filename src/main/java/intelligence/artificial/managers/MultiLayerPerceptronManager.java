package intelligence.artificial.managers;

import intelligence.artificial.logic.MultiLayerPerceptron;
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

    public static void trainModel(MultiLayerNetwork model, DataSet trainData, int epochs, int batchSize) {
        MultiLayerPerceptron.trainModel(model, trainData, epochs, batchSize);
    }

    public static void setWeights(MultiLayerNetwork model, int nIn, int layerIndex, double[] weights) {
        MultiLayerPerceptron.setWeights(model, nIn, layerIndex, weights);
    }

    public static void saveModel(MultiLayerNetwork model, String filePath) throws IOException {
        ModelSerializer.writeModel(model, new File(filePath), true);
    }

    public static MultiLayerNetwork loadModel(String filePath) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
    }
}
