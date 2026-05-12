import os
import tempfile
import unittest

from PyQt5.QtGui import QImage

from object_annotator.libs.assistant.dataset import DatasetBuildConfig, build_yolo_dataset


def _write_image(path, width=128, height=96):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(0)
    ok = image.save(path)
    if not ok:
        raise RuntimeError("Failed to save test image: %s" % path)


class TestAssistantDataset(unittest.TestCase):
    def test_build_yolo_dataset_from_existing_yolo_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_dir = os.path.join(tmp, "images")
            labels_dir = os.path.join(tmp, "labels")
            output_dir = os.path.join(tmp, "out_dataset")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            with open(
                os.path.join(labels_dir, "classes.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("cat\n")
                f.write("dog\n")

            # Create five images, first four labeled.
            for i in range(1, 6):
                image_name = "img%d.jpg" % i
                _write_image(os.path.join(images_dir, image_name))

            for i in range(1, 5):
                label_name = "img%d.txt" % i
                with open(
                    os.path.join(labels_dir, label_name), "w", encoding="utf-8"
                ) as f:
                    if i % 2 == 0:
                        f.write("1 0.500000 0.500000 0.400000 0.400000\n")
                    else:
                        f.write("0 0.500000 0.500000 0.400000 0.400000\n")

            result = build_yolo_dataset(
                DatasetBuildConfig(
                    image_dir=images_dir,
                    annotation_dir=labels_dir,
                    output_dir=output_dir,
                    max_images=3,
                    val_ratio=0.2,
                    seed=7,
                )
            )

            self.assertEqual(result.selected_images, 3)
            self.assertEqual(result.train_images + result.val_images, 3)
            self.assertGreaterEqual(result.train_images, 1)
            self.assertGreaterEqual(result.val_images, 1)
            self.assertTrue(os.path.isfile(result.dataset_yaml_path))

            self.assertTrue(os.path.isdir(os.path.join(output_dir, "images", "train")))
            self.assertTrue(os.path.isdir(os.path.join(output_dir, "images", "val")))
            self.assertTrue(os.path.isdir(os.path.join(output_dir, "labels", "train")))
            self.assertTrue(os.path.isdir(os.path.join(output_dir, "labels", "val")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "classes.txt")))

            with open(
                os.path.join(output_dir, "classes.txt"), "r", encoding="utf-8"
            ) as f:
                classes = [line.strip() for line in f if line.strip()]
            self.assertTrue("cat" in classes)

    def test_build_dataset_requires_two_labeled_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_dir = os.path.join(tmp, "images")
            labels_dir = os.path.join(tmp, "labels")
            output_dir = os.path.join(tmp, "out_dataset")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            _write_image(os.path.join(images_dir, "img1.jpg"))

            with open(
                os.path.join(labels_dir, "classes.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("cat\n")

            with open(os.path.join(labels_dir, "img1.txt"), "w", encoding="utf-8") as f:
                f.write("0 0.500000 0.500000 0.400000 0.400000\n")

            with self.assertRaises(ValueError):
                build_yolo_dataset(
                    DatasetBuildConfig(
                        image_dir=images_dir,
                        annotation_dir=labels_dir,
                        output_dir=output_dir,
                        max_images=100,
                    )
                )

    def test_build_dataset_rerun_ignores_output_dir_inside_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_dir = os.path.join(tmp, "images")
            labels_dir = os.path.join(tmp, "labels")
            output_dir = os.path.join(images_dir, ".assistant", "dataset")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            with open(
                os.path.join(labels_dir, "classes.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("cat\n")

            for i in range(1, 4):
                image_name = "img%d.jpg" % i
                _write_image(os.path.join(images_dir, image_name))
                with open(
                    os.path.join(labels_dir, "img%d.txt" % i), "w", encoding="utf-8"
                ) as f:
                    f.write("0 0.500000 0.500000 0.400000 0.400000\n")

            # First run should create output under image dir.
            first = build_yolo_dataset(
                DatasetBuildConfig(
                    image_dir=images_dir,
                    annotation_dir=labels_dir,
                    output_dir=output_dir,
                    max_images=3,
                )
            )
            self.assertEqual(first.selected_images, 3)

            # Second run should not recurse into prior output or self-copy.
            second = build_yolo_dataset(
                DatasetBuildConfig(
                    image_dir=images_dir,
                    annotation_dir=labels_dir,
                    output_dir=output_dir,
                    max_images=3,
                )
            )
            self.assertEqual(second.selected_images, 3)


if __name__ == "__main__":
    unittest.main()
