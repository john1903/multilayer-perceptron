package intelligence.artificial.logic;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;


public class EpochScoreListener {

    private int epochCounter = 0;
    private final INDArray[] trainingData;
    private final INDArray[] testData;

    public EpochScoreListener(INDArray[] trainingData, INDArray[] testData) {
        this.trainingData = trainingData;
        this.testData = testData;
    }

    public void onEpochEnd(MultiLayerNetwork model) {
        double trainingMSE = calculateMSE(model, trainingData);
        double testMSE = calculateMSE(model, testData);
        epochCounter++;
        System.out.println(epochCounter + "," + trainingMSE + "," + testMSE);
    }

    private double calculateMSE(MultiLayerNetwork model, INDArray[] data) {
        INDArray predictedOutputs = model.output(data[0]);

        INDArray diff = predictedOutputs.sub(data[1]);
        INDArray squaredDiff = diff.mul(diff);
        return squaredDiff.meanNumber().doubleValue();
    }
}
