package intelligence.artificial;

import intelligence.artificial.logic.EpochScoreListener;
import intelligence.artificial.managers.DataManager;
import intelligence.artificial.managers.MultiLayerPerceptronManager;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.FileWriter;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        if (args.length < 1) {
            printUsage();
            return;
        }

        MultiLayerPerceptronManager manager;
        MultiLayerNetwork model;

        try {
            switch (args[0]) {
                case "create":
                    if (args.length < 4) {
                        printUsage();
                        return;
                    }
                    String[] hiddenLayersStr = args[1].split(",");
                    int[] hiddenLayers = new int[hiddenLayersStr.length];
                    for (int i = 0; i < hiddenLayersStr.length; i++) {
                        hiddenLayers[i] = Integer.parseInt(hiddenLayersStr[i].trim());
                    }
                    double learningRate = Double.parseDouble(args[2]);
                    String savePath = args[3];

                    manager = new MultiLayerPerceptronManager(hiddenLayers, learningRate);
                    model = manager.createModel();
                    MultiLayerPerceptronManager.saveModel(model, savePath);
                    System.out.println("Model created and saved successfully.");
                    break;

                case "train":
                    if (args.length < 5) {
                        printUsage();
                        return;
                    }
                    String modelPath = args[1];
                    String trainDataPath = args[2];
                    String testDataPath = args[3];
                    int epochs = Integer.parseInt(args[4]);

                    model = MultiLayerPerceptronManager.loadModel(modelPath);
                    DataManager dataManager = new DataManager(trainDataPath);
                    INDArray[] data = dataManager.loadAllData();
                    if (data == null || data.length < 2 || data[0].rows() == 0 || data[1].rows() == 0) {
                        System.out.println("Invalid data. Skipping training.");
                        return;
                    }
                    dataManager = new DataManager(testDataPath);
                    INDArray[] testData = dataManager.loadAllData();
                    if (testData == null || testData.length < 2 || testData[0].rows() == 0 || testData[1].rows() == 0) {
                        System.out.println("Invalid test data. Skipping training.");
                        return;
                    }
                    DataSet trainData = new DataSet(data[0], data[1]);
                    EpochScoreListener epochScoreListener = new EpochScoreListener(data, testData);
                    for (int i = 0; i < epochs; i++) {
                        MultiLayerPerceptronManager.trainModel(model, trainData, 10);
                        epochScoreListener.onEpochEnd(model);
                    }
                    MultiLayerPerceptronManager.saveModel(model, modelPath);
                    System.out.println("Model trained and saved successfully.");
                    break;

                case "set-weights":
                    if (args.length < 6) {
                        printUsage();
                        return;
                    }
                    modelPath = args[1];
                    int layerIndex = Integer.parseInt(args[2]);
                    int nIn = Integer.parseInt(args[3]);
                    String[] weightsStr = args[4].split(",");
                    double[] weights = new double[weightsStr.length];
                    for (int i = 0; i < weightsStr.length; i++) {
                        weights[i] = Double.parseDouble(weightsStr[i].trim());
                    }

                    model = MultiLayerPerceptronManager.loadModel(modelPath);
                    MultiLayerPerceptronManager.setWeights(model, nIn, layerIndex, weights);
                    MultiLayerPerceptronManager.saveModel(model, modelPath);
                    System.out.println("Weights set and model saved successfully.");
                    break;

                case "predict":
                    if (args.length < 4) {
                        printUsage();
                        return;
                    }
                    modelPath = args[1];
                    String userDataPath = args[2];
                    String csvFilePath = args[3];

                    model = MultiLayerPerceptronManager.loadModel(modelPath);
                    generatePredictionsCSV(model, userDataPath, csvFilePath);
                    System.out.println("CSV file generated successfully.");
                    break;

                default:
                    printUsage();
                    break;
            }
        } catch (IOException e) {
            System.out.println("An error occurred: " + e.getMessage());
        } catch (NumberFormatException e) {
            System.out.println("Invalid number format: " + e.getMessage());
        }
    }

    private static void generatePredictionsCSV(MultiLayerNetwork model, String userDataPath, String csvFilePath) throws IOException {
        DataManager dataManager = new DataManager(userDataPath);
        INDArray[] data = dataManager.loadAllData();

        INDArray inputs = data[0];
        INDArray actualOutputs = data[1];
        INDArray predictedOutputs = model.output(inputs);

        try (FileWriter writer = new FileWriter(csvFilePath)) {
            writer.append("inputX,inputY,predictedPositionX,predictedPositionY,actualX,actualY\n");

            for (int i = 0; i < inputs.rows(); i++) {
                writer.append(String.valueOf(inputs.getFloat(i, 0))).append(",")
                        .append(String.valueOf(inputs.getFloat(i, 1))).append(",")
                        .append(String.valueOf(predictedOutputs.getFloat(i, 0))).append(",")
                        .append(String.valueOf(predictedOutputs.getFloat(i, 1))).append(",")
                        .append(String.valueOf(actualOutputs.getFloat(i, 0))).append(",")
                        .append(String.valueOf(actualOutputs.getFloat(i, 1))).append("\n");
            }
        }
    }

    private static void printUsage() {
        System.out.println("Usage:");
        System.out.println("  create <hidden_layers> <learning_rate> <path_to_save>");
        System.out.println("  train <model_path> <training_data_path> <test_data_path> <epochs>");
        System.out.println("  set-weights <model_path> <layer_index> <n_in> <weights>");
        System.out.println("  predict <model_path> <user_data_path> <csv_file_path>");
    }
}