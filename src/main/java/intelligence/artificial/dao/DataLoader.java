package intelligence.artificial.dao;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class DataLoader implements DataLoaderInterface {
    @Override
    public double[][][] loadData(String filePath) throws IOException {

        List<String> lines = Files.readAllLines(Paths.get(filePath));
        List<double[]> inputDataList = new ArrayList<>();
        List<double[]> outputDataList = new ArrayList<>();

        for (String line : lines) {
            String[] values = line.split(",");
            try {
                double input1 = Double.parseDouble(values[0]);
                double input2 = Double.parseDouble(values[1]);
                double output1 = Double.parseDouble(values[2]);
                double output2 = Double.parseDouble(values[3]);

                inputDataList.add(new double[]{input1, input2});
                outputDataList.add(new double[]{output1, output2});
            } catch (NumberFormatException e) {
                System.out.println("Skipping line: " + line + " in file: " + filePath);
            }
        }

        double[][] inputData = new double[inputDataList.size()][2];
        double[][] outputData = new double[outputDataList.size()][2];

        for (int i = 0; i < inputDataList.size(); i++) {
            inputData[i] = inputDataList.get(i);
        }

        for (int i = 0; i < outputDataList.size(); i++) {
            outputData[i] = outputDataList.get(i);
        }

        return new double[][][]{inputData, outputData};
    }
}