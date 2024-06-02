package intelligence.artificial.dao;

import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DataLoaderTest {

    @Test
    public void testLoadData() throws IOException {
        DataLoader dataLoader = new DataLoader();
        String filePath = "src/test/resources/test.csv";
        double[][][] data = dataLoader.loadData(filePath);

        double[][] expectedInputs = new double[][]{
                {1.0, 4.0},
                {2.0, 5.0},
                {3.0, 6.0}
        };

        double[][] expectedOutputs = new double[][]{
                {7.0, 10.0},
                {8.0, 11.0},
                {9.0, 12.0}
        };

        assertEquals(expectedInputs.length, data[0].length);
        assertEquals(expectedInputs[0].length, data[0][0].length);
        assertEquals(expectedOutputs.length, data[1].length);
        assertEquals(expectedOutputs[0].length, data[1][0].length);

        assertArrayEquals(expectedInputs, data[0]);
        assertArrayEquals(expectedOutputs, data[1]);
    }
}
