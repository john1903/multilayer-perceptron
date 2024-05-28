package intelligence.artificial.dao;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class DataLoader implements DataLoaderInterface {
    @Override
    public INDArray[] loadData(String filePath) throws IOException {

        List<String> lines = Files.readAllLines(Paths.get(filePath));
        double[][] inputData = new double[2][lines.size()];
        double[][] outputData = new double[2][lines.size()];

        for (int i = 0; i < lines.size(); i++) {
            String[] values = lines.get(i).split(",");
            inputData[0][i] = Double.parseDouble(values[0]);
            inputData[1][i] = Double.parseDouble(values[1]);
            outputData[0][i] = Double.parseDouble(values[2]);
            outputData[1][i] = Double.parseDouble(values[3]);
        }

        INDArray inputs = Nd4j.createFromArray(inputData).transpose();
        INDArray outputs = Nd4j.createFromArray(outputData).transpose();

        return new INDArray[]{inputs, outputs};
    }
}