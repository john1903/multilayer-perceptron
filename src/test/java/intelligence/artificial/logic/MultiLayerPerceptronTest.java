package intelligence.artificial.logic;

import static org.junit.jupiter.api.Assertions.*;

import intelligence.artificial.managers.DataManager;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class MultiLayerPerceptronTest {

    @Test
    public void testCreateModel() {
        int[] hiddenLayers = {3, 3};
        double learningRate = 0.01;

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(hiddenLayers, learningRate);
        MultiLayerNetwork model = mlp.createModel();

        assertNotNull(model);

        int expectedNumLayers = hiddenLayers.length + 2;
        assertEquals(expectedNumLayers, model.getnLayers());

        Layer[] layers = model.getLayers();
        assertEquals(2, layers[0].getParam("W").columns());
        assertEquals(3, layers[1].getParam("W").columns());
        assertEquals(3, layers[2].getParam("W").columns());
        assertEquals(2, layers[3].getParam("W").columns());
    }

    @Test
    public void testSetWeights() {
        int[] hiddenLayers = {3, 3};
        double learningRate = 0.01;

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(hiddenLayers, learningRate);
        MultiLayerNetwork model = mlp.createModel();

        double[] weights = {0.1, 0.2, 0.3};
        int nIn = 2;
        int layerIndex = 1;

        MultiLayerPerceptron.setWeights(model, nIn, layerIndex, weights);
        INDArray weightArray = model.getLayer(layerIndex).getParam("W");
        double[] actualWeights = new double[weights.length];
        int count = 0;
        for (int i = 0; i < 6; i++) {
            if (i % 2 == 0) {
                actualWeights[count] = weightArray.getDouble(i);
                count++;
            }
        }
        double scalingFactor = Math.sqrt(2.0 / (nIn + weights.length));
        INDArray expected = Nd4j.create(weights).muli(scalingFactor);
        double[] expectedWeights = expected.data().asDouble();
        assertArrayEquals(expectedWeights, actualWeights, 1e-6);

    }

    @Test
    public void testTrainModel() throws IOException {
        int[] hiddenLayers = {4, 4};
        double learningRate = 0.0001;

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(hiddenLayers, learningRate);
        MultiLayerNetwork model = mlp.createModel();

        DataManager dataManager = new DataManager("src/test/resources/");
        INDArray[] data = dataManager.loadAllData();

        INDArray inputData = data[0];
        INDArray outputData = data[1];

        assertDoesNotThrow(() -> MultiLayerPerceptron.trainModel(model, new DataSet(inputData, outputData), 10));
    }
}