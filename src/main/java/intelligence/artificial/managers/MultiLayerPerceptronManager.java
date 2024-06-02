package intelligence.artificial.managers;

import intelligence.artificial.logic.MultiLayerPerceptron;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

public class MultiLayerPerceptronManager {
    private final MultiLayerPerceptron multiLayerPerceptron;
    public MultiLayerPerceptronManager(int[] hiddenLayers, double learningRate, int epochs) {
        multiLayerPerceptron = new MultiLayerPerceptron(hiddenLayers, learningRate, epochs);
    }

    public MultiLayerPerceptronManager(MultiLayerNetwork model) {
        multiLayerPerceptron = new MultiLayerPerceptron(model);
    }

    public MultiLayerNetwork createModel() {
        return multiLayerPerceptron.createModel();
    }

    public void trainModel(MultiLayerNetwork model, DataSetIterator trainData) {
        multiLayerPerceptron.trainModel(model, trainData);
    }

    public void setWeights(MultiLayerNetwork model, int nIn, int layerIndex, double[] weights) {
        multiLayerPerceptron.setWeights(model, nIn, layerIndex, weights);
    }

    public static void saveModel(MultiLayerNetwork model, String filePath) throws IOException {
        ModelSerializer.writeModel(model, new File(filePath), true);
    }

    public static MultiLayerNetwork loadModel(String filePath) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(new File(filePath));
    }
}
