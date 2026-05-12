# Object Annotator

Object Annotator is a desktop tool for labeling images for object detection datasets. It supports drawing bounding boxes and exporting annotations in Pascal VOC, YOLO, and CreateML formats.

## Install

```bash
pip install object-annotator
```

To enable the optional YOLO training/prediction assistant:

```bash
pip install "object-annotator[assistant]"
```

## Run

```bash
object-annotator
```

Optional startup arguments:

```bash
object-annotator [IMAGE_DIR] [CLASS_FILE] [SAVE_DIR]
```

## Features

- Draw, edit, copy, and delete bounding boxes
- Label objects with custom classes
- Export Pascal VOC (`.xml`), YOLO (`.txt`), and CreateML (`.json`)
- Quickly navigate image folders
- Optional YOLO-assisted prediction and training workflow

## Common shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl+U` | Open image directory |
| `Ctrl+R` | Change save directory |
| `Ctrl+S` | Save annotation |
| `Ctrl+D` | Duplicate selected box |
| `W` | Create bounding box |
| `A` | Previous image |
| `D` | Next image |
| `Delete` | Delete selected box |

## Development

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m unittest discover tests
python -m build
```

The application package lives in `src/object_annotator`.
