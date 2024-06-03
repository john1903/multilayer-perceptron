package intelligence.artificial.logic;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultiLayerPerceptron {
    private final int[] hiddenLayers;
    private final double learningRate;

    public MultiLayerPerceptron(int[] hiddenLayers, double learningRate) {
        this.hiddenLayers = hiddenLayers;
        this.learningRate = learningRate;
    }

    public MultiLayerNetwork createModel() {
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .l2(0.001)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
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

        return model;
    }

    public static void setWeights(MultiLayerNetwork model, int nIn, int layerIndex, double[] weights) {
        INDArray weightArray = Nd4j.create(weights);

        int nOut = weights.length;
        double scalingFactor = Math.sqrt(2.0 / (nIn + nOut));
        weightArray.muli(scalingFactor);

        model.getLayer(layerIndex).setParam("W", weightArray);
    }

    public static void trainModel(MultiLayerNetwork model, DataSet dataSet, int epochs, int batchSize) {
        model.setListeners(new EpochScoreListener());
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);

        for (int i = 0; i < epochs; i++) {
            dataSet.shuffle();

            DataSetIterator shuffledTrainData = new ListDataSetIterator<>(dataSet.asList(), batchSize);

            while (shuffledTrainData.hasNext()) {
                DataSet batch = shuffledTrainData.next();
                normalizer.transform(batch);
                model.fit(batch);
            }

            shuffledTrainData.reset();
            model.getListeners().forEach(listener -> listener.onEpochEnd(model));
        }
    }
}
