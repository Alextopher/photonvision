package org.photonvision.vision.objects;

import java.awt.Color;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.ColorHelper;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

public class OpenCVObjectDetector implements ObjectDetector {
    private static final Logger logger =
            new Logger(OpenCVObjectDetector.class, LogGroup.VisionModule);

    private final OpenCVModel model;
    private final Size inputSize;

    private final Net net;

    public OpenCVObjectDetector(OpenCVModel model, Size inputSize) {
        this.model = model;
        this.inputSize = inputSize;

        // Read the file into a MatOfByte
        MatOfByte buffer = new MatOfByte();
        try {
            byte[] bytes = Files.readAllBytes(Paths.get(model.modelFile.getPath()));
            buffer.fromArray(bytes);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read model file " + model.modelFile.getPath(), e);
        }

        // Load the model
        net = Dnn.readNetFromONNX(buffer);
    }

    @Override
    public void release() {
        // TODO: 🙏 pray that the GC does it's job and calls Net.finalize() in a timely manner.
    }

    @Override
    public Model getModel() {
        return model;
    }

    @Override
    public List<String> getClasses() {
        return model.labels;
    }

    /**
     * Detects objects in the given input image using the RknnDetector.
     *
     * @param in The input image to perform object detection on.
     * @param nmsThresh The threshold value for non-maximum suppression.
     * @param confidence The threshold value for bounding box detection.
     * @return A list of NeuralNetworkPipeResult objects representing the detected objects. Returns an
     *     empty list if the detector is not initialized or if no objects are detected.
     */
    @Override
    public List<NeuralNetworkPipeResult> detect(Mat in, double nmsThresh, double confidence) {
        Mat letterboxed = new Mat();
        Letterbox scale =
                Letterbox.letterbox(in, letterboxed, this.inputSize, ColorHelper.colorToScalar(Color.GRAY));

        if (!letterboxed.size().equals(this.inputSize)) {
            throw new RuntimeException("Letterboxed frame is not the right size!");
        }

        switch (model.family) {
            case YOLO:
                detectYolo(letterboxed, nmsThresh, confidence);
                break;
        }

        return new ArrayList<>();
    }

    public List<NeuralNetworkPipeResult> detectYolo(Mat in, double nmsThresh, double confidence) {
        net.setInput(in);

        // [ [x_center, y_center, width, height, confidence, class_id], ... ]
        Mat output = net.forward();

        logger.info("Output shape: " + output.size().toString());

        // @param bboxes a set of bounding boxes to apply NMS.
        // @param scores a set of corresponding confidences.
        // @param class_ids a set of corresponding class ids. Ids are integer and usually start from 0.
        // @param score_threshold a threshold used to filter boxes by score.
        // @param nms_threshold a threshold used in non maximum suppression.
        // @param indices the kept indices of bboxes after NMS.

        // MatOfRect2d bboxes;
        // MatOfFloat scores;
        // MatOfInt class_ids;
        // float score_threshold;
        // float nms_threshold;
        // MatOfInt indices;

        // Dnn.NMSBoxesBatched(null, null, null, 0, 0, null);

        return new ArrayList<>();
    }
}
