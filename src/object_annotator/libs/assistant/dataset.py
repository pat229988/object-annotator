import os
import random
import shutil
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtGui import QImage

from ..create_ml_io import CreateMLReader
from ..labelFile import LabelFile
from ..pascal_voc_io import PascalVocReader
from ..utils import natural_sort


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".pbm",
    ".pgm",
    ".ppm",
    ".sr",
    ".ras",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class DatasetBuildConfig:
    image_dir: str
    output_dir: str
    annotation_dir: Optional[str] = None
    max_images: int = 100
    val_ratio: float = 0.1
    seed: int = 42


@dataclass
class DatasetBuildResult:
    selected_images: int
    train_images: int
    val_images: int
    classes: list
    dataset_yaml_path: str
    skipped_images: int
    total_labeled_available: int


def scan_images(image_dir, excluded_roots=None):
    excluded_roots = [
        os.path.abspath(path)
        for path in (excluded_roots or [])
        if path and os.path.isdir(path)
    ]
    images = []
    for root, _, files in os.walk(image_dir):
        abs_root = os.path.abspath(root)
        if any(
            abs_root == excluded or abs_root.startswith(excluded + os.sep)
            for excluded in excluded_roots
        ):
            continue
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(os.path.abspath(os.path.join(root, file_name)))
    natural_sort(images, key=lambda x: x.lower())
    return images


def resolve_annotation_path(image_path, annotation_dir: Optional[str] = None):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    parent = annotation_dir if annotation_dir else os.path.dirname(image_path)

    xml_path = os.path.join(parent, base_name + ".xml")
    txt_path = os.path.join(parent, base_name + ".txt")
    json_path = os.path.join(parent, base_name + ".json")

    if os.path.isfile(xml_path):
        return xml_path
    if os.path.isfile(txt_path):
        return txt_path
    if os.path.isfile(json_path):
        return json_path
    return None


def read_shapes(annotation_path, image_path):
    ext = os.path.splitext(annotation_path)[1].lower()

    if ext == ".xml":
        return PascalVocReader(annotation_path).get_shapes()

    if ext == ".txt":
        image = QImage()
        image.load(image_path)
        if image.isNull():
            return []
        classes_path = os.path.join(os.path.dirname(annotation_path), "classes.txt")
        if not os.path.isfile(classes_path):
            return []
        return _read_yolo_shapes(
            annotation_path, image.width(), image.height(), classes_path
        )

    if ext == ".json":
        return CreateMLReader(annotation_path, image_path).get_shapes()

    return []


def _read_yolo_shapes(annotation_path, width, height, classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    shapes = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_index, x_center, y_center, box_width, box_height = parts
            try:
                class_index = int(class_index)
                x_center = float(x_center)
                y_center = float(y_center)
                box_width = float(box_width)
                box_height = float(box_height)
            except (TypeError, ValueError):
                continue

            if class_index < 0 or class_index >= len(classes):
                continue

            label = classes[class_index]

            x_min = max(x_center - box_width / 2.0, 0.0)
            x_max = min(x_center + box_width / 2.0, 1.0)
            y_min = max(y_center - box_height / 2.0, 0.0)
            y_max = min(y_center + box_height / 2.0, 1.0)

            x_min = round(width * x_min)
            x_max = round(width * x_max)
            y_min = round(height * y_min)
            y_max = round(height * y_max)

            points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            shapes.append((label, points, None, None, False))

    return shapes


def _bbox_to_yolo_line(points, width, height):
    x_min, y_min, x_max, y_max = LabelFile.convert_points_to_bnd_box(points)

    x_center = float(x_min + x_max) / 2.0 / float(width)
    y_center = float(y_min + y_max) / 2.0 / float(height)
    box_width = float(x_max - x_min) / float(width)
    box_height = float(y_max - y_min) / float(height)

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    box_width = min(max(box_width, 0.0), 1.0)
    box_height = min(max(box_height, 0.0), 1.0)

    return x_center, y_center, box_width, box_height


def _split_images(images, val_ratio, seed):
    if len(images) <= 1:
        return images, []

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_ratio))
    if val_count <= 0:
        val_count = 1
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1

    val_images = shuffled[:val_count]
    train_images = shuffled[val_count:]
    return train_images, val_images


def _ensure_structure(output_dir):
    images_train = os.path.join(output_dir, "images", "train")
    images_val = os.path.join(output_dir, "images", "val")
    labels_train = os.path.join(output_dir, "labels", "train")
    labels_val = os.path.join(output_dir, "labels", "val")

    os.makedirs(images_train, exist_ok=True)
    os.makedirs(images_val, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    return {
        "images_train": images_train,
        "images_val": images_val,
        "labels_train": labels_train,
        "labels_val": labels_val,
    }


def build_yolo_dataset(config):
    if not config.image_dir or not os.path.isdir(config.image_dir):
        raise ValueError("image_dir is missing or invalid")

    effective_max = int(config.max_images)
    if effective_max < 0:
        raise ValueError("max_images must be 0 (all) or greater")

    output_abs = os.path.abspath(config.output_dir)
    image_dir_abs = os.path.abspath(config.image_dir)

    # Avoid stale files and self-copy problems on re-runs.
    if os.path.isdir(output_abs):
        shutil.rmtree(output_abs)

    all_images = scan_images(image_dir_abs, excluded_roots=[output_abs])
    labeled = []
    total_labeled_available = 0
    skipped = 0

    for image_path in all_images:
        annotation_path = resolve_annotation_path(image_path, config.annotation_dir)
        if not annotation_path:
            continue

        shapes = read_shapes(annotation_path, image_path)
        if not shapes:
            skipped += 1
            continue

        total_labeled_available += 1
        if effective_max == 0 or len(labeled) < effective_max:
            labeled.append((image_path, annotation_path, shapes))

    if len(labeled) < 2:
        raise ValueError("Need at least 2 labeled images to build a train/val dataset")

    selected_images = [item[0] for item in labeled]
    train_images, val_images = _split_images(
        selected_images, config.val_ratio, config.seed
    )

    train_set = set(train_images)
    val_set = set(val_images)

    folders = _ensure_structure(output_abs)

    class_names = []
    class_to_index = {}

    for image_path, _annotation_path, shapes in labeled:
        image = QImage()
        image.load(image_path)
        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            skipped += 1
            continue

        if image_path in train_set:
            image_out_dir = folders["images_train"]
            label_out_dir = folders["labels_train"]
        elif image_path in val_set:
            image_out_dir = folders["images_val"]
            label_out_dir = folders["labels_val"]
        else:
            # Should never happen, but keep stable behavior.
            continue

        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        image_target = os.path.join(image_out_dir, image_name)
        label_target = os.path.join(label_out_dir, label_name)

        if os.path.abspath(image_path) != os.path.abspath(image_target):
            shutil.copy2(image_path, image_target)

        lines = []
        for label, points, _line_color, _fill_color, _difficult in shapes:
            if label not in class_to_index:
                class_to_index[label] = len(class_names)
                class_names.append(label)

            class_index = class_to_index[label]
            x_center, y_center, box_width, box_height = _bbox_to_yolo_line(
                points, image.width(), image.height()
            )
            lines.append(
                "%d %.6f %.6f %.6f %.6f"
                % (class_index, x_center, y_center, box_width, box_height)
            )

        with open(label_target, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    classes_path = os.path.join(output_abs, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    dataset_yaml_path = os.path.join(output_abs, "dataset.yaml")
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % output_abs)
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for index, name in enumerate(class_names):
            f.write("  %d: %s\n" % (index, name))

    return DatasetBuildResult(
        selected_images=len(labeled),
        train_images=len(train_images),
        val_images=len(val_images),
        classes=class_names,
        dataset_yaml_path=dataset_yaml_path,
        skipped_images=skipped,
        total_labeled_available=total_labeled_available,
    )
