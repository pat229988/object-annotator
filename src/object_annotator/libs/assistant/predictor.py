import os


def sanitize_xyxy(x1, y1, x2, y2, width, height):
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(x1, max(width - 1, 0)))
    y1 = max(0, min(y1, max(height - 1, 0)))
    x2 = max(0, min(x2, max(width - 1, 0)))
    y2 = max(0, min(y2, max(height - 1, 0)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 == x1:
        x2 = min(x1 + 1, max(width - 1, 0))
    if y2 == y1:
        y2 = min(y1 + 1, max(height - 1, 0))

    return x1, y1, x2, y2


class YoloPredictor:
    def __init__(self):
        self._model = None
        self._model_path = ""

    def clear_model_cache(self):
        self._model = None
        self._model_path = ""

    def _load_model(self, model_path):
        if not model_path or not os.path.isfile(model_path):
            raise ValueError("Model checkpoint not found: %s" % model_path)

        if self._model is not None and self._model_path == model_path:
            return self._model

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics is not available: %s" % e)

        self._model = YOLO(model_path)
        self._model_path = model_path
        return self._model

    def predict(
        self, model_path, image_path, conf=0.35, iou=0.5, device="cpu", imgsz=512
    ):
        model = self._load_model(model_path)

        if not image_path or not os.path.isfile(image_path):
            raise ValueError("Image not found: %s" % image_path)

        results = model.predict(
            source=image_path,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        image_h = int(getattr(result, "orig_shape", [0, 0])[0] or 0)
        image_w = int(getattr(result, "orig_shape", [0, 0])[1] or 0)

        output = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v.item()) for v in box.xyxy[0]]
            x1, y1, x2, y2 = sanitize_xyxy(x1, y1, x2, y2, image_w, image_h)

            label = names.get(cls_id, str(cls_id))
            output.append(
                {
                    "label": label,
                    "confidence": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        return output
