import unittest

from object_annotator.libs.assistant.predictor import sanitize_xyxy
from object_annotator.libs.assistant.predictor import YoloPredictor


class TestAssistantPredictorHelpers(unittest.TestCase):
    def test_sanitize_xyxy_clips_and_orders_box(self):
        x1, y1, x2, y2 = sanitize_xyxy(120.8, 80.2, -10.7, 15.6, 100, 90)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 99)
        self.assertLessEqual(y2, 89)
        self.assertGreaterEqual(x2, x1)
        self.assertGreaterEqual(y2, y1)

    def test_sanitize_xyxy_expands_zero_area(self):
        x1, y1, x2, y2 = sanitize_xyxy(10, 10, 10, 10, 50, 50)
        self.assertNotEqual(x1, x2)
        self.assertNotEqual(y1, y2)

    def test_predictor_cache_can_be_cleared(self):
        predictor = YoloPredictor()
        predictor._model = object()
        predictor._model_path = "/tmp/fake.pt"
        predictor.clear_model_cache()
        self.assertIsNone(predictor._model)
        self.assertEqual(predictor._model_path, "")


if __name__ == "__main__":
    unittest.main()
