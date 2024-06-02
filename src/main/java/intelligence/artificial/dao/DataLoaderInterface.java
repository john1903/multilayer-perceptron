package intelligence.artificial.dao;

import java.io.IOException;

public interface DataLoaderInterface {
    double[][][] loadData(String filePath) throws IOException;
}
