package org.photonvision.vision.objects;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import org.opencv.core.Size;

public class OpenCVModel implements Model {
    public final File modelFile;
    public final ModelFamily family;
    public final List<String> labels;
    public final Size inputSize;

    public enum ModelFamily {
        YOLO,
    }

    /**
     * OpenCV DNN model constructor.
     *
     * @param modelFile path to model on disk. Format: `name-width-height-model.onnx`
     * @param labels path to labels file on disk
     * @throws IllegalArgumentException
     */
    public OpenCVModel(File modelFile, String labels) {
        this.modelFile = modelFile;

        String[] parts = modelFile.getName().split("-");
        if (parts.length != 4) {
            throw new IllegalArgumentException("Invalid model file name: " + modelFile);
        }

        int width = Integer.parseInt(parts[1]);
        int height = Integer.parseInt(parts[2]);
        this.inputSize = new Size(width, height);

        if (parts[3].toLowerCase().contains("yolo")) {
            this.family = ModelFamily.YOLO;
        } else {
            throw new IllegalArgumentException("Unknown ModelFamily for model " + modelFile);
        }

        try {
            this.labels = Files.readAllLines(Paths.get(labels));
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read labels file " + labels, e);
        }
    }

    @Override
    public ObjectDetector load() {
        return new OpenCVObjectDetector(this, inputSize);
    }

    @Override
    public String getName() {
        return modelFile.getName();
    }
}
