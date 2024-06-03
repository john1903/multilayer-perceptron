package intelligence.artificial.logic;

import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.nn.api.Model;

import java.io.Serializable;

public class EpochScoreListener extends BaseTrainingListener implements Serializable {

    private int epochCounter = 0;

    @Override
    public void onEpochEnd(Model model) {
        double lastScore = model.score();
        epochCounter++;
        System.out.println("Epoch: " + epochCounter + " - Model Score: " + lastScore);
    }
}
