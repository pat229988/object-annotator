import unittest

from object_annotator.libs.assistant.device import resolve_device
from object_annotator.libs.assistant.trainer import build_train_command, parse_progress_line


class TestAssistantTrainerHelpers(unittest.TestCase):
    def test_parse_progress_line_detects_epoch_pattern(self):
        parsed = parse_progress_line("  3/10    1.23G   box_loss")
        self.assertEqual(parsed, (3, 10))

    def test_parse_progress_line_returns_none_for_non_progress(self):
        self.assertIsNone(parse_progress_line("Results saved to runs/detect/mini"))

    def test_build_train_command_contains_expected_tokens(self):
        args, best = build_train_command(
            dataset_yaml_path="/tmp/dataset.yaml",
            runs_dir="/tmp/runs",
            epochs=10,
            imgsz=512,
            device="cpu",
        )

        self.assertTrue("-c" in args)
        self.assertTrue("from ultralytics import YOLO" in args[1])
        self.assertTrue("10" in args)
        self.assertTrue("512" in args)
        self.assertTrue("cpu" in args)
        self.assertTrue("2" in args)
        self.assertTrue("False" in args)
        self.assertTrue(best.endswith("/mini/weights/best.pt"))

    def test_build_train_command_cuda_uses_amp_and_larger_batch(self):
        args, _best = build_train_command(
            dataset_yaml_path="/tmp/dataset.yaml",
            runs_dir="/tmp/runs",
            epochs=10,
            imgsz=512,
            device="cuda",
        )
        self.assertTrue("4" in args)
        self.assertTrue("True" in args)

    def test_resolve_device_returns_known_value(self):
        value = resolve_device()
        self.assertTrue(value in {"cuda", "mps", "cpu"})


if __name__ == "__main__":
    unittest.main()
