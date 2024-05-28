package intelligence.artificial.dao;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public interface DataLoaderInterface {
    INDArray[] loadData(String filePath) throws IOException;
}
