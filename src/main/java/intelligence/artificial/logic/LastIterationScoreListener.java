package intelligence.artificial.logic;

import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.nn.api.Model;

import java.io.Serializable;

public class LastIterationScoreListener extends BaseTrainingListener implements Serializable {
    private final int printIterations;

    public LastIterationScoreListener(int printIterations) {
        this.printIterations = printIterations;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        double lastScore = model.score();
        if (iteration % printIterations == 0) {
            System.out.println("MODEL_SCORE: " + lastScore);
        }
    }
}