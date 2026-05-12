import os
import tempfile
from unittest import TestCase

from PyQt5.QtGui import QImage

from object_annotator.app import get_main_app


class _RunningTrainer:
    def is_running(self):
        return True

    def stop_training(self):
        return None


class TestAssistantMainWindowFlow(TestCase):
    app = None
    win = None

    def setUp(self):
        self.app, self.win = get_main_app()

    def tearDown(self):
        self.win.close()
        self.app.quit()

    def _write_image(self, path):
        image = QImage(120, 90, QImage.Format_RGB32)
        image.fill(0)
        ok = image.save(path)
        self.assertTrue(ok)

    def test_auto_predict_sets_pending_for_unlabeled_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "img1.jpg")
            model_path = os.path.join(tmp, "model.pt")
            self._write_image(image_path)
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("fake")

            self.win.file_path = image_path
            self.win.default_save_dir = None
            self.win.assistant_enabled = True
            self.win.assistant_auto_predict = True
            self.win.assistant_is_training = False
            self.win.assistant_is_predicting = False
            self.win.assistant_last_model_path = model_path
            self.win.assistant_last_auto_predict_path = ""
            self.win.assistant_auto_predict_pending_path = ""

            self.win.maybe_auto_predict_loaded_image()

            self.assertEqual(self.win.assistant_auto_predict_pending_path, image_path)

    def test_auto_predict_skips_when_annotation_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "img1.jpg")
            model_path = os.path.join(tmp, "model.pt")
            xml_path = os.path.splitext(image_path)[0] + ".xml"

            self._write_image(image_path)
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("fake")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write("<annotation/>")

            self.win.file_path = image_path
            self.win.default_save_dir = None
            self.win.assistant_enabled = True
            self.win.assistant_auto_predict = True
            self.win.assistant_last_model_path = model_path
            self.win.assistant_auto_predict_pending_path = ""

            self.win.maybe_auto_predict_loaded_image()

            self.assertEqual(self.win.assistant_auto_predict_pending_path, "")

    def test_predict_skips_when_training_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "img1.jpg")
            model_path = os.path.join(tmp, "model.pt")
            self._write_image(image_path)
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("fake")

            self.win.file_path = image_path
            self.win.default_save_dir = None
            self.win.assistant_enabled = True
            self.win.assistant_is_predicting = False
            self.win.assistant_last_model_path = model_path
            self.win.assistant_trainer = _RunningTrainer()

            self.win._predict_current_image(interactive=False)

            self.assertFalse(self.win.assistant_is_predicting)


if __name__ == "__main__":
    import unittest

    unittest.main()
