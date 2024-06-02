package intelligence.artificial;

import intelligence.artificial.managers.DataManager;
import intelligence.artificial.managers.MultiLayerPerceptronManager;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        MultiLayerPerceptronManager manager = null;
        MultiLayerNetwork model = null;

        while (true) {
            System.out.println("Select an option:");
            System.out.println("1. Create a new model");
            System.out.println("2. Train model");
            System.out.println("3. Set weights");
            System.out.println("4. Save model");
            System.out.println("5. Load model");
            System.out.println("6. Generate CSV with predictions");
            System.out.println("7. Exit");

            String choiceStr = scanner.nextLine();
            int choice;
            try {
                choice = Integer.parseInt(choiceStr);
            } catch (NumberFormatException e) {
                System.out.println("Invalid option. Please enter a number.");
                continue;
            }

            try {
                switch (choice) {
                    case 1:
                        System.out.print("Enter hidden layers (comma separated): ");
                        String[] hiddenLayersStr = scanner.nextLine().split(",");
                        int[] hiddenLayers = new int[hiddenLayersStr.length];
                        for (int i = 0; i < hiddenLayersStr.length; i++) {
                            hiddenLayers[i] = Integer.parseInt(hiddenLayersStr[i].trim());
                        }
                        System.out.print("Enter learning rate: ");
                        double learningRate = Double.parseDouble(scanner.nextLine());
                        System.out.print("Enter epochs: ");
                        int epochs = Integer.parseInt(scanner.nextLine());

                        manager = new MultiLayerPerceptronManager(hiddenLayers, learningRate, epochs);
                        model = manager.createModel();
                        System.out.println("Model created successfully.");
                        break;

                    case 2:
                        if (model == null) {
                            System.out.println("Create or load a model first.");
                            break;
                        }
                        System.out.print("Enter the path to the training data folder: ");
                        String trainDataPath = scanner.nextLine();
                        System.out.println(trainDataPath);
                        if (trainDataPath.isEmpty()) {
                            System.out.println("Path cannot be empty. Please try again.");
                            break;
                        }
                        DataManager dataManager = new DataManager(trainDataPath);
                        List<INDArray[]> data = dataManager.loadAllData();
                        for (INDArray[] indArray : data) {
                            if (indArray == null || indArray.length < 2 || indArray[0].rows() == 0 || indArray[1].rows() == 0) {
                                System.out.println("Invalid data. Please check the data files and try again.");
                                continue;
                            }
                            DataSetIterator trainData = new ListDataSetIterator<>(List.of(new DataSet(indArray[0], indArray[1])), 10);
                            manager.trainModel(model, trainData);
                        }
                        System.out.println("Model trained successfully.");
                        break;

                    case 3:
                        if (model == null) {
                            System.out.println("Create or load a model first.");
                            break;
                        }
                        System.out.print("Enter layer index: ");
                        int layerIndex = Integer.parseInt(scanner.nextLine());
                        System.out.print("Enter number of inputs: ");
                        int nIn = Integer.parseInt(scanner.nextLine());
                        System.out.print("Enter weights (comma separated): ");
                        String[] weightsStr = scanner.nextLine().split(",");
                        double[] weights = new double[weightsStr.length];
                        for (int i = 0; i < weightsStr.length; i++) {
                            weights[i] = Double.parseDouble(weightsStr[i].trim());
                        }
                        manager.setWeights(model, nIn, layerIndex, weights);
                        System.out.println("Weights set successfully.");
                        break;

                    case 4:
                        if (model == null) {
                            System.out.println("Create or load a model first.");
                            break;
                        }
                        System.out.print("Enter file path to save the model: ");
                        String savePath = scanner.nextLine();
                        manager.saveModel(model, savePath);
                        System.out.println("Model saved successfully.");
                        break;

                    case 5:
                        System.out.print("Enter file path to load the model: ");
                        String loadPath = scanner.nextLine();
                        assert manager != null;
                        model = manager.loadModel(loadPath);
                        System.out.println("Model loaded successfully.");
                        break;

                    case 6:
                        if (model == null) {
                            System.out.println("Create or load a model first.");
                            break;
                        }
                        System.out.print("Enter the path to the user data file: ");
                        String userDataPath = scanner.nextLine();
                        System.out.print("Enter the path to save the CSV file: ");
                        String csvFilePath = scanner.nextLine();
                        generatePredictionsCSV(model, userDataPath, csvFilePath);
                        System.out.println("CSV file generated successfully.");
                        break;

                    case 7:
                        System.out.println("Exiting...");
                        return;

                    default:
                        System.out.println("Invalid option. Please try again.");
                }
            } catch (IOException e) {
                System.out.println("An error occurred: " + e.getMessage());
            } catch (NumberFormatException e) {
                System.out.println("Invalid number format: " + e.getMessage());
            }
        }
    }

    private static void generatePredictionsCSV(MultiLayerNetwork model, String userDataPath, String csvFilePath) throws IOException {
        DataManager dataManager = new DataManager(userDataPath);
        List<INDArray[]> data = dataManager.loadAllData();

        try (FileWriter writer = new FileWriter(csvFilePath)) {
            writer.append("inputX,inputY,calculatedPositionX,calculatedPositionY,actualX,actualY\n");

            for (INDArray[] indArray : data) {
                if (indArray == null || indArray.length < 2 || indArray[0].rows() == 0 || indArray[1].rows() == 0) {
                    System.out.println("Invalid data. Skipping entry.");
                    continue;
                }

                INDArray inputs = indArray[0];
                INDArray actualOutputs = indArray[1];
                INDArray predictedOutputs = model.output(inputs);

                for (int i = 0; i < inputs.rows(); i++) {
                    writer.append(String.valueOf(inputs.getFloat(i, 0))).append(",").append(String.valueOf(inputs.getFloat(i, 1))).append(",");
                    writer.append(String.valueOf(predictedOutputs.getFloat(i, 0))).append(",").append(String.valueOf(predictedOutputs.getFloat(i, 1))).append(",");
                    writer.append(String.valueOf(actualOutputs.getFloat(i, 0))).append(",").append(String.valueOf(actualOutputs.getFloat(i, 1))).append("\n");
                }
            }
        }
    }
}
