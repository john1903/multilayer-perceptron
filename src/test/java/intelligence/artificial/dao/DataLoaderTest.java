package intelligence.artificial.dao;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DataLoaderTest {

    @Test
    public void testLoadData() throws IOException {
        DataLoader dataLoader = new DataLoader();
        String filePath = "src/test/resources/test.csv";
        INDArray[] data = dataLoader.loadData(filePath);

        INDArray expectedInputs = Nd4j.create(new double[][]{
                {1.0, 4.0},
                {2.0, 5.0},
                {3.0, 6.0}
        });

        INDArray expectedOutputs = Nd4j.create(new double[][]{
                {7.0, 10.0},
                {8.0, 11.0},
                {9.0, 12.0}
        });

        assertEquals(expectedInputs.shape()[0], data[0].shape()[0]);
        assertEquals(expectedInputs.shape()[1], data[0].shape()[1]);
        assertEquals(expectedOutputs.shape()[0], data[1].shape()[0]);
        assertEquals(expectedOutputs.shape()[1], data[1].shape()[1]);

        assertArrayEquals(expectedInputs.toDoubleMatrix(), data[0].toDoubleMatrix());
        assertArrayEquals(expectedOutputs.toDoubleMatrix(), data[1].toDoubleMatrix());
    }
}
