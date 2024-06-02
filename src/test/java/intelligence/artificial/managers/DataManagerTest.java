package intelligence.artificial.managers;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DataManagerTest {

    private Path testDir;
    private DataManager dataManager;

    @BeforeEach
    public void setUp() throws IOException {
        testDir = Files.createTempDirectory("dataManagerTest");

        createTestDataFile(testDir.resolve("file1.csv"), new double[][]{{1.0, 2.0}, {3.0, 4.0}}, new double[][]{{5.0, 6.0}, {7.0, 8.0}});
        createTestDataFile(testDir.resolve("file2.csv"), new double[][]{{9.0, 10.0}, {11.0, 12.0}}, new double[][]{{13.0, 14.0}, {15.0, 16.0}});

        dataManager = new DataManager(testDir.toString());
    }

    @AfterEach
    public void tearDown() throws IOException {
        Files.walkFileTree(testDir, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.delete(dir);
                return FileVisitResult.CONTINUE;
            }
        });
    }

    @Test
    public void testLoadAllData() throws IOException {
        List<INDArray[]> allData = dataManager.loadAllData();

        INDArray[] expectedDataFile1 = new INDArray[]{
                Nd4j.create(new double[][]{{1.0, 2.0}, {3.0, 4.0}}),
                Nd4j.create(new double[][]{{5.0, 6.0}, {7.0, 8.0}})
        };
        INDArray[] expectedDataFile2 = new INDArray[]{
                Nd4j.create(new double[][]{{9.0, 10.0}, {11.0, 12.0}}),
                Nd4j.create(new double[][]{{13.0, 14.0}, {15.0, 16.0}})
        };

        assertEquals(2, allData.size());
        assertArrayEquals(expectedDataFile1, allData.get(0));
        assertArrayEquals(expectedDataFile2, allData.get(1));
    }

    private void createTestDataFile(Path filePath, double[][] inputs, double[][] outputs) throws IOException {
        List<String> lines = List.of(
                inputs[0][0] + "," + inputs[0][1] + "," + outputs[0][0] + "," + outputs[0][1],
                inputs[1][0] + "," + inputs[1][1] + "," + outputs[1][0] + "," + outputs[1][1]
        );
        Files.write(filePath, lines, StandardOpenOption.CREATE);
    }
}