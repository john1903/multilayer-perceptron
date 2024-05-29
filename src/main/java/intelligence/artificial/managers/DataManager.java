package intelligence.artificial.managers;

import intelligence.artificial.dao.DataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

public class DataManager {
    private final String folderPath;
    private final DataLoader dataLoader;

    public DataManager(String folderPath) {
        this.folderPath = folderPath;
        this.dataLoader = new DataLoader();
    }

    public INDArray[] loadAllData() throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();

        DirectoryStream<Path> directoryStream = Files.newDirectoryStream(Paths.get(folderPath));

        for (Path path : directoryStream) {
            if (Files.isRegularFile(path)) {
                INDArray[] data = dataLoader.loadData(path.toString());
                inputList.add(data[0]);
                outputList.add(data[1]);
            }
        }

        INDArray combinedInputs = Nd4j.vstack(inputList);
        INDArray combinedOutputs = Nd4j.vstack(outputList);

        return new INDArray[]{combinedInputs, combinedOutputs};
    }
}
