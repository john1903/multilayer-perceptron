package intelligence.artificial.managers;

import intelligence.artificial.dao.DataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

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

    public List<INDArray[]> loadAllData() throws IOException {
        List<INDArray[]> loadedData = new ArrayList<>();

        DirectoryStream<Path> directoryStream = Files.newDirectoryStream(Paths.get(folderPath));

        for (Path path : directoryStream) {
            if (Files.isRegularFile(path)) {
                INDArray[] data = dataLoader.loadData(path.toString());
                loadedData.add(data);
            }
        }

        return loadedData;
    }
}
