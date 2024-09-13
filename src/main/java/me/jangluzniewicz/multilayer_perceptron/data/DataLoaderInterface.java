package me.jangluzniewicz.multilayer_perceptron.data;

import java.io.IOException;

public interface DataLoaderInterface {
    double[][][] loadData(String filePath) throws IOException;
}
