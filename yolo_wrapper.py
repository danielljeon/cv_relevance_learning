import cv2
import numpy as np
import onnxruntime as ort


class Detection:
    def __init__(self, class_id, x, y, w, h, conf):
        # All coords normalized 0-1 relative to input size.
        self.class_id = class_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf


class YOLO:
    def __init__(self, model_path="yolov5n.onnx", input_size=640):
        self.session = ort.InferenceSession(model_path)
        self.input_size = input_size

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, frame):
        # Letterbox-ish resize without preserving aspect ratio.
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        img = img.astype(np.float16)

        preds = self.session.run([self.output_name], {self.input_name: img})[0]

        # YOLOv5 ONNX output: (1, N, 85) -> (N, 85).
        preds = preds[0]

        detections = []

        for det in preds:
            obj_conf = float(det[4])
            if obj_conf < 0.15:  # Was 0.25 - loosened.
                continue

            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])

            overall_conf = obj_conf * class_conf
            if overall_conf < 0.30:  # Was 0.5 - loosened a lot.
                continue

            x, y, w, h = det[0], det[1], det[2], det[3]

            detections.append(
                Detection(
                    class_id=class_id,
                    x=x / self.input_size,
                    y=y / self.input_size,
                    w=w / self.input_size,
                    h=h / self.input_size,
                    conf=overall_conf,
                )
            )

        return detections
