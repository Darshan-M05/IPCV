import sys
import os

# CRITICAL: import torch BEFORE PyQt5 on Windows.
# PyQt5 loads DLLs that conflict with PyTorch's CUDA libraries (c10.dll).
# Loading torch first avoids the "DLL initialization routine failed" error.
import torch  # noqa: F401  (side-effect import)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QImage

# Ensure src/ is on the path so worker can import siblings
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from worker import VideoWorker


# ─── Dark‑themed stylesheet ──────────────────────────────────────────────────
_STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QLabel {
    color: #cdd6f4;
}
QLabel#title {
    color: #89b4fa;
    font-size: 20px;
    font-weight: bold;
}
QLabel#videoDisplay {
    background-color: #11111b;
    border: 2px solid #313244;
    border-radius: 8px;
}
QGroupBox {
    color: #a6adc8;
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 18px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: bold;
    min-width: 140px;
}
QPushButton:hover {
    background-color: #45475a;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border: 1px solid #313244;
}
QPushButton#runBtn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
}
QPushButton#runBtn:hover {
    background-color: #94e2d5;
}
QPushButton#runBtn:disabled {
    background-color: #313244;
    color: #585b70;
}
QPushButton#stopBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
}
QPushButton#stopBtn:hover {
    background-color: #eba0ac;
}
QPushButton#stopBtn:disabled {
    background-color: #313244;
    color: #585b70;
}
QPushButton#exitBtn {
    background-color: #fab387;
    color: #1e1e2e;
    border: none;
}
QPushButton#exitBtn:hover {
    background-color: #f9e2af;
}
QLabel#statValue {
    color: #a6e3a1;
    font-size: 14px;
    font-weight: bold;
}
QLabel#statLabel {
    color: #6c7086;
    font-size: 11px;
}
"""


class MainWindow(QMainWindow):
    """Professional GUI for the object tracking pipeline."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Moving Object Detection & Trajectory Tracking")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(_STYLESHEET)

        self._video_path = None
        self._worker = None

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._default_dir = os.path.join(base_dir, "dataset", "VIRAT")
        self._output_dir = os.path.join(base_dir, "output")

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ── Left: video display ──────────────────────────────────────────
        left = QVBoxLayout()

        title = QLabel("Moving Object Detection & Trajectory Tracking")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        left.addWidget(title)

        self.video_label = QLabel()
        self.video_label.setObjectName("videoDisplay")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setText("No video loaded")
        left.addWidget(self.video_label, 1)

        root.addLayout(left, 3)

        # ── Right: controls + stats ──────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        # Controls group
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(8)

        self.select_btn = QPushButton("Select Video")
        self.select_btn.clicked.connect(self._on_select)
        ctrl_layout.addWidget(self.select_btn)

        self.run_btn = QPushButton("Run Model")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        ctrl_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self.stop_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setObjectName("exitBtn")
        self.exit_btn.clicked.connect(self.close)
        ctrl_layout.addWidget(self.exit_btn)

        ctrl_box.setLayout(ctrl_layout)
        right.addWidget(ctrl_box)

        # Video info group
        info_box = QGroupBox("Video")
        info_layout = QVBoxLayout()
        self.video_name_label = QLabel("None selected")
        self.video_name_label.setWordWrap(True)
        info_layout.addWidget(self.video_name_label)
        info_box.setLayout(info_layout)
        right.addWidget(info_box)

        # Statistics group
        stats_box = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(10)

        self.stat_model = self._make_stat_row("Model", "-")
        stats_layout.addLayout(self.stat_model[0])

        self.stat_fps = self._make_stat_row("FPS", "-")
        stats_layout.addLayout(self.stat_fps[0])

        self.stat_frame = self._make_stat_row("Frame", "-")
        stats_layout.addLayout(self.stat_frame[0])

        self.stat_tracks = self._make_stat_row("Active Tracks", "-")
        stats_layout.addLayout(self.stat_tracks[0])

        self.stat_dets = self._make_stat_row("Detections", "-")
        stats_layout.addLayout(self.stat_dets[0])

        stats_box.setLayout(stats_layout)
        right.addWidget(stats_box)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #6c7086; font-style: italic;")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)

        right.addStretch()
        root.addLayout(right, 1)

    @staticmethod
    def _make_stat_row(label_text, default):
        layout = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("statLabel")
        val = QLabel(default)
        val.setObjectName("statValue")
        val.setAlignment(Qt.AlignRight)
        layout.addWidget(lbl)
        layout.addWidget(val)
        return layout, val

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_select(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", self._default_dir,
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)",
        )
        if path:
            self._video_path = path
            self.video_name_label.setText(os.path.basename(path))
            self.run_btn.setEnabled(True)
            self.status_label.setText("Video selected - press Run Model")
            self.video_label.setText("Press Run Model to start")

    def _on_run(self):
        if not self._video_path:
            return

        self.select_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Initializing...")

        self._worker = VideoWorker(
            video_path=self._video_path,
            output_dir=self._output_dir,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self._on_status)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.status_label.setText("Stopping...")

    def _on_status(self, msg):
        """Status update from the worker (e.g. 'Loading model...')."""
        self.status_label.setText(msg)

    def _on_frame(self, qimg, stats):
        # Scale image to fit the display label while keeping aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

        # Update stats
        self.stat_model[1].setText(str(stats.get("model", "-")))
        self.stat_fps[1].setText(str(stats.get("fps", "-")))
        self.stat_frame[1].setText(str(stats.get("frame", "-")))
        self.stat_tracks[1].setText(str(stats.get("tracks", "-")))
        self.stat_dets[1].setText(str(stats.get("detections", "-")))
        self.status_label.setText("Processing...")

    def _on_finished(self, output_path):
        self.select_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Done - saved to {os.path.basename(output_path)}")

    def _on_error(self, msg):
        self.select_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Pipeline Error", msg)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(5000)
        event.accept()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
