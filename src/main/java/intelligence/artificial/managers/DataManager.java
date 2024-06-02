package intelligence.artificial.managers;

import intelligence.artificial.dao.DataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataManager {
    private final String folderPath;
    private final DataLoader dataLoader;

    public DataManager(String folderPath) {
        this.folderPath = folderPath;
        this.dataLoader = new DataLoader();
    }

    public INDArray[] loadAllData() throws IOException {
        List<double[]> allInputDataList = new ArrayList<>();
        List<double[]> allOutputDataList = new ArrayList<>();

        try (DirectoryStream<Path> directoryStream = Files.newDirectoryStream(Paths.get(folderPath))) {
            for (Path path : directoryStream) {
                if (Files.isRegularFile(path)) {
                    double[][][] data = dataLoader.loadData(path.toString());
                    allInputDataList.addAll(Arrays.asList(data[0]));
                    allOutputDataList.addAll(Arrays.asList(data[1]));
                }
            }
        }

        double[][] allInputData = allInputDataList.toArray(new double[0][0]);
        double[][] allOutputData = allOutputDataList.toArray(new double[0][0]);

        INDArray inputDataINDArray = Nd4j.create(allInputData);
        INDArray outputDataINDArray = Nd4j.create(allOutputData);

        return new INDArray[]{inputDataINDArray, outputDataINDArray};
    }
}
