import os
import re
import sys

from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal


def parse_progress_line(line):
    """Parse common epoch progress patterns from trainer output."""
    match = re.search(r"(?:^|\s)(\d+)\s*/\s*(\d+)(?:\s|$)", line)
    if not match:
        return None

    epoch = int(match.group(1))
    total = int(match.group(2))
    if total <= 0:
        return None
    if epoch < 0:
        return None
    return epoch, total


def build_train_command(
    dataset_yaml_path,
    runs_dir,
    epochs,
    imgsz,
    device,
    model_name="yolov8n.pt",
):
    device = (device or "cpu").strip().lower()
    batch = 2 if device == "cpu" else 4
    amp = "True" if device == "cuda" else "False"

    name = "mini"
    train_code = (
        "import sys; "
        "from ultralytics import YOLO; "
        "data,project,epochs,imgsz,device,model,batch,amp,name = sys.argv[1:10]; "
        "YOLO(model).train("
        "task='detect', mode='train', "
        "exist_ok=True, verbose=True, workers=0, "
        "batch=int(batch), patience=10, save=True, save_period=1, "
        "plots=False, amp=(amp=='True'), cache=False, deterministic=True, seed=42, "
        "name=name, data=data, project=project, epochs=int(epochs), imgsz=int(imgsz), device=device"
        ")"
    )
    args = [
        "-c",
        train_code,
        dataset_yaml_path,
        runs_dir,
        str(int(epochs)),
        str(int(imgsz)),
        device,
        model_name,
        str(batch),
        amp,
        name,
    ]
    expected_best = os.path.join(runs_dir, name, "weights", "best.pt")
    return args, expected_best


class YoloTrainer(QObject):
    started = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    message = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)
    stopped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self._stopped_by_user = False
        self._expected_best = ""
        self._log_lines = []

    def is_running(self):
        return self.process is not None and self.process.state() != QProcess.NotRunning

    def start_training(
        self,
        dataset_yaml_path,
        runs_dir,
        epochs,
        imgsz,
        device,
        model_name="yolov8n.pt",
    ):
        if self.is_running():
            raise RuntimeError("Training is already running")

        if not os.path.isfile(dataset_yaml_path):
            raise ValueError("Dataset YAML not found: %s" % dataset_yaml_path)

        os.makedirs(runs_dir, exist_ok=True)

        args, expected_best = build_train_command(
            dataset_yaml_path=dataset_yaml_path,
            runs_dir=runs_dir,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            model_name=model_name,
        )

        self._expected_best = expected_best
        self._stopped_by_user = False
        self._log_lines = []

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(args)
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("OMP_NUM_THREADS", "2")
        env.insert("MKL_NUM_THREADS", "2")
        self.process.setProcessEnvironment(env)

        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

        self.started.emit(
            "Training started on %s. Output: %s"
            % (device, os.path.dirname(expected_best))
        )
        self.process.start()

    def stop_training(self):
        if not self.is_running():
            return

        self._stopped_by_user = True
        self.process.terminate()

        def _force_kill():
            if self.is_running():
                self.process.kill()

        QTimer.singleShot(2500, _force_kill)

    def _append_logs(self, text):
        if not text:
            return
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            self._log_lines.append(line)
            if len(self._log_lines) > 250:
                self._log_lines = self._log_lines[-250:]
            self.message.emit(line)

            progress = parse_progress_line(line)
            if progress:
                self.progress.emit(progress[0], progress[1], line)

    def _on_output(self):
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput())
        text = raw.decode("utf-8", errors="ignore")
        self._append_logs(text)

    def _on_error(self, _error):
        if self.process is None:
            return
        # Error signal is informative; final status handled by _on_finished.

    def _on_finished(self, exit_code, _exit_status):
        if self.process is not None:
            raw = bytes(self.process.readAllStandardOutput())
            if raw:
                self._append_logs(raw.decode("utf-8", errors="ignore"))

        if self._stopped_by_user:
            self.stopped.emit("Training stopped by user")
        elif exit_code == 0 and os.path.isfile(self._expected_best):
            self.completed.emit(self._expected_best)
        elif exit_code == 0:
            self.failed.emit(
                "Training finished but best checkpoint was not found at %s"
                % self._expected_best
            )
        else:
            tail = "\n".join(self._log_lines[-10:])
            self.failed.emit(
                "Training failed with exit code %d. Recent output:\n%s"
                % (exit_code, tail)
            )

        self.process = None
