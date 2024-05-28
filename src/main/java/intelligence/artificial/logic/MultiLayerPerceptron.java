package intelligence.artificial.logic;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultiLayerPerceptron {
    private final int[] hiddenLayers;
    private final double learningRate;
    private final int epochs;

    public MultiLayerPerceptron(int[] hiddenLayers, double learningRate, int epochs) {
        this.hiddenLayers = hiddenLayers;
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    public MultiLayerNetwork createModel() {
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .list();

        listBuilder.layer(0, new DenseLayer.Builder().nIn(2).nOut(2)
                .activation(Activation.RELU).weightInit(WeightInit.XAVIER).build());

        for (int i = 0; i < hiddenLayers.length; i++) {
            listBuilder.layer(i + 1, new DenseLayer.Builder().nIn(i == 0 ? 2 : hiddenLayers[i - 1])
                    .nOut(hiddenLayers[i]).activation(Activation.RELU).weightInit(WeightInit.XAVIER).build());
        }

        listBuilder.layer(hiddenLayers.length + 1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(hiddenLayers[hiddenLayers.length - 1]).nOut(2).build());

        MultiLayerConfiguration conf = listBuilder.build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        return model;
    }

    public void setWeights(MultiLayerNetwork model, int nIn, int layerIndex, double[] weights) {
        INDArray weightArray = Nd4j.create(weights);

        int nOut = weights.length;
        double scalingFactor = Math.sqrt(2.0 / (nIn + nOut));
        weightArray.muli(scalingFactor);

        model.getLayer(layerIndex).setParam("W", weightArray);
    }

    public void trainModel(MultiLayerNetwork model, DataSetIterator trainData) {
        for (int i = 0; i < epochs; i++) {
            model.fit(trainData);
        }
    }
}
