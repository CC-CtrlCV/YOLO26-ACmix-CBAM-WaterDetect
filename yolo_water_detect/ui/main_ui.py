from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_water_detect.detect.yolo_infer import YoloInferencer
from yolo_water_detect.utils.export_tool import ExportManager
from yolo_water_detect.utils.qss_style import APP_QSS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


class DetectionWorker(QThread):
    frame_ready = pyqtSignal(object, list, float, int, int)
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str, str)
    stats = pyqtSignal(dict)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, inferencer: YoloInferencer, config: dict):
        super().__init__()
        self.inferencer = inferencer
        self.config = config
        self._paused = False
        self._stopped = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            mode = self.config["mode"]
            if mode == "单张图片":
                self._process_images([Path(self.config["source"])])
            elif mode == "图片文件夹":
                folder = Path(self.config["source"])
                files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
                self._process_images(files)
            elif mode == "本地视频文件":
                self._process_video(self.config["source"])
            else:
                self._process_stream(self.config["source"])
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _wait_if_paused(self):
        while self._paused and not self._stopped:
            time.sleep(0.05)

    def _infer(self, frame, frame_index=0, source_name=""):
        annotated, detections, elapsed = self.inferencer.infer_frame(
            frame,
            conf=self.config["conf"],
            iou=self.config["iou"],
            imgsz=self.config["imgsz"],
            device=self.config["device"],
            half=self.config["half"],
            line_width=self.config["line_width"],
            font_scale=self.config["font_size"] / 18.0,
            show_conf=self.config["show_conf"],
            grid=self.config["grid_crop"],
            show_grid=self.config["show_grid"],
            tracker=self.config["tracker"],
        )
        exporter = self.config["exporter"]
        stem = f"{Path(source_name).stem}_{frame_index:06d}" if frame_index else Path(source_name).stem
        if self.config["save_image"]:
            exporter.save_image(annotated, stem)
        if self.config["save_crops"]:
            exporter.save_crops(frame, detections, stem)
        if self.config["save_json"]:
            exporter.save_json({"source": source_name, "frame": frame_index, "detections": detections}, stem)
        if self.config["save_csv"]:
            exporter.append_csv(source_name, frame_index, detections)
        return annotated, detections, elapsed

    def _process_images(self, files):
        max_items = self.config.get("max_items", 0)
        if max_items > 0:
            files = files[:max_items]
        total = max(1, len(files))
        all_counts = Counter()
        start = time.perf_counter()
        for i, path in enumerate(files, 1):
            if self._stopped:
                break
            self._wait_if_paused()
            frame = cv2.imread(str(path))
            if frame is None:
                self.log.emit(f"跳过无法读取的图片：{path}", "warn")
                continue
            annotated, detections, elapsed = self._infer(frame, 0, str(path))
            all_counts.update([d["class_name"] for d in detections])
            self.frame_ready.emit(annotated, detections, elapsed, i, total)
            self.progress.emit(i, total)
            self.stats.emit(self._stats(detections, all_counts, start))
            self.log.emit(f"完成图片 {i}/{total}: {path.name}，目标 {len(detections)} 个，耗时 {elapsed:.1f} ms", "info")

    def _process_video(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = None
        if self.config["save_video"]:
            out_path = self.config["exporter"].video_dir / f"{Path(path).stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        all_counts = Counter()
        frame_idx = 0
        done = 0
        start = time.perf_counter()
        skip = max(1, self.config["frame_skip"])
        while not self._stopped:
            self._wait_if_paused()
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % skip != 0:
                continue
            done += 1
            annotated, detections, elapsed = self._infer(frame, frame_idx, str(path))
            if writer:
                writer.write(annotated)
            all_counts.update([d["class_name"] for d in detections])
            self.frame_ready.emit(annotated, detections, elapsed, frame_idx, total)
            self.progress.emit(min(frame_idx, total), total)
            self.stats.emit(self._stats(detections, all_counts, start))
        cap.release()
        if writer:
            writer.release()

    def _process_stream(self, source):
        src = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头或网络流：{source}")
        all_counts = Counter()
        start = time.perf_counter()
        frame_idx = 0
        while not self._stopped:
            self._wait_if_paused()
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % max(1, self.config["frame_skip"]) != 0:
                continue
            annotated, detections, elapsed = self._infer(frame, frame_idx, str(source))
            all_counts.update([d["class_name"] for d in detections])
            self.frame_ready.emit(annotated, detections, elapsed, frame_idx, 0)
            self.stats.emit(self._stats(detections, all_counts, start))
        cap.release()

    @staticmethod
    def _stats(current, total_counts, start):
        sizes = []
        for det in current:
            x1, y1, x2, y2 = det["xyxy"]
            sizes.append(max(0, x2 - x1) * max(0, y2 - y1))
        elapsed = max(0.001, time.perf_counter() - start)
        return {
            "current_count": len(current),
            "total_count": int(sum(total_counts.values())),
            "class_counts": dict(total_counts),
            "max_size": int(max(sizes) if sizes else 0),
            "min_size": int(min(sizes) if sizes else 0),
            "elapsed": elapsed,
            "avg_fps": int(sum(total_counts.values()) / elapsed) if elapsed else 0,
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水面漂浮物智能检测系统 V1.0")
        self.resize(1440, 900)
        self.setStyleSheet(APP_QSS)
        self.inferencer = YoloInferencer()
        self.worker = None
        self.last_pixmap = None
        self._build_ui()
        self.scan_weights()
        self.log("系统就绪，请先加载模型。", "info")

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._left_panel())
        splitter.addWidget(self._center_panel())
        splitter.addWidget(self._right_panel())
        splitter.setSizes([360, 760, 320])
        layout.addWidget(splitter, stretch=1)

        self.statusBar().showMessage("数据源：未选择 | 模型：未加载 | 设备：CPU")
        self.setCentralWidget(root)

    def _header(self):
        frame = QFrame()
        frame.setObjectName("Header")
        layout = QHBoxLayout(frame)
        self.title = QLabel("🌊 水面漂浮物智能检测系统 V1.0")
        self.title.setObjectName("TitleLabel")
        self.status_dot = QLabel()
        self.status_dot.setObjectName("StatusDotIdle")
        self.status_text = QLabel("空闲")
        self.fps_label = self._metric("FPS", "0")
        self.time_label = self._metric("耗时(ms)", "0")
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("PrimaryButton")
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("终止")
        self.stop_btn.setObjectName("DangerButton")
        self.export_btn = QPushButton("打开输出目录")
        self.start_btn.clicked.connect(self.start_detection)
        self.pause_btn.clicked.connect(self.pause_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.export_btn.clicked.connect(self.open_output_dir)
        layout.addWidget(self.title)
        layout.addStretch(1)
        layout.addWidget(self.status_dot)
        layout.addWidget(self.status_text)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.time_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.export_btn)
        return frame

    def _metric(self, name, value):
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 0, 8, 0)
        val = QLabel(value)
        val.setObjectName("MetricValue")
        lab = QLabel(name)
        lab.setObjectName("MetricName")
        lay.addWidget(val, alignment=Qt.AlignCenter)
        lay.addWidget(lab, alignment=Qt.AlignCenter)
        box.value_label = val
        return box

    def _left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.addWidget(self._model_card())
        lay.addWidget(self._source_card())
        lay.addWidget(self._visual_card())
        lay.addWidget(self._output_card())
        lay.addWidget(self._advanced_card())
        lay.addStretch(1)
        scroll.setWidget(wrap)
        scroll.setFixedWidth(380)
        return scroll

    def _card(self, title):
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        label = QLabel(title)
        label.setObjectName("CardTitle")
        lay.addWidget(label)
        return card, lay

    def _model_card(self):
        card, lay = self._card("1. 模型加载")
        self.model_path = QLineEdit("yolo26n.pt")
        self.model_combo = QComboBox()
        browse = QPushButton("浏览")
        load = QPushButton("加载模型")
        load.setObjectName("PrimaryButton")
        browse.clicked.connect(self.browse_model)
        load.clicked.connect(self.load_model)
        self.device_combo = QComboBox(); self.device_combo.addItems(["cpu", "0", "1"])
        self.conf_slider = self._slider(1, 99, 25)
        self.iou_slider = self._slider(10, 90, 45)
        self.imgsz_combo = QComboBox(); self.imgsz_combo.addItems(["640", "800", "1280", "1920"]); self.imgsz_combo.setCurrentText("800")
        lay.addLayout(self._row("模型路径", self.model_path, browse))
        lay.addWidget(QLabel("权重快速切换")); lay.addWidget(self.model_combo)
        lay.addLayout(self._row("设备", self.device_combo))
        lay.addLayout(self._row("置信度", self.conf_slider))
        lay.addLayout(self._row("NMS IoU", self.iou_slider))
        lay.addLayout(self._row("输入尺寸", self.imgsz_combo))
        lay.addWidget(load)
        self.model_combo.currentTextChanged.connect(lambda p: self.model_path.setText(p) if p else None)
        return card

    def _source_card(self):
        card, lay = self._card("2. 数据源输入")
        self.mode_combo = QComboBox(); self.mode_combo.addItems(["单张图片", "图片文件夹", "本地视频文件", "摄像头/RTSP实时流"])
        self.source_path = QLineEdit()
        browse = QPushButton("选择")
        browse.clicked.connect(self.browse_source)
        self.camera_input = QLineEdit("0")
        self.grid_crop = QCheckBox("启用视频/图像分块裁剪推理（4×6，25%重叠）")
        lay.addLayout(self._row("输入模式", self.mode_combo))
        lay.addLayout(self._row("文件/目录", self.source_path, browse))
        lay.addLayout(self._row("摄像头/RTSP", self.camera_input))
        lay.addWidget(self.grid_crop)
        return card

    def _visual_card(self):
        card, lay = self._card("3. 可视化参数")
        self.line_slider = self._slider(1, 8, 2)
        self.font_slider = self._slider(8, 32, 14)
        self.show_conf = QCheckBox("显示置信度"); self.show_conf.setChecked(True)
        self.show_id = QCheckBox("显示目标ID")
        self.show_grid = QCheckBox("显示网格分割线")
        self.save_crops = QCheckBox("保存裁剪目标图")
        lay.addLayout(self._row("框线粗细", self.line_slider))
        lay.addLayout(self._row("文字字号", self.font_slider))
        for w in (self.show_conf, self.show_id, self.show_grid, self.save_crops):
            lay.addWidget(w)
        return card

    def _output_card(self):
        card, lay = self._card("4. 批量输出")
        self.output_dir = QLineEdit("output_dataset")
        browse = QPushButton("选择")
        browse.clicked.connect(self.browse_output)
        self.save_image = QCheckBox("保存带框图片"); self.save_image.setChecked(True)
        self.save_json = QCheckBox("导出JSON元数据"); self.save_json.setChecked(True)
        self.save_csv = QCheckBox("导出CSV统计"); self.save_csv.setChecked(True)
        self.save_video = QCheckBox("导出标注视频")
        self.max_items = QSpinBox(); self.max_items.setRange(0, 999999); self.max_items.setValue(0)
        lay.addLayout(self._row("输出目录", self.output_dir, browse))
        for w in (self.save_image, self.save_crops, self.save_json, self.save_csv, self.save_video):
            lay.addWidget(w)
        lay.addLayout(self._row("最大处理数", self.max_items))
        return card

    def _advanced_card(self):
        card, lay = self._card("5. 高级推理")
        self.thread_num = QSpinBox(); self.thread_num.setRange(1, 16); self.thread_num.setValue(1)
        self.fp16 = QCheckBox("启用FP16加速（GPU）")
        self.frame_skip = QSpinBox(); self.frame_skip.setRange(1, 999); self.frame_skip.setValue(1)
        self.tracker = QCheckBox("启用ByteTrack多目标跟踪")
        lay.addLayout(self._row("线程数", self.thread_num))
        lay.addWidget(self.fp16)
        lay.addLayout(self._row("跳帧间隔", self.frame_skip))
        lay.addWidget(self.tracker)
        return card

    def _center_panel(self):
        box = QWidget()
        lay = QVBoxLayout(box)
        self.viewer = QLabel("请选择模型和数据源后开始检测")
        self.viewer.setAlignment(Qt.AlignCenter)
        self.viewer.setMinimumSize(640, 480)
        self.viewer.setStyleSheet("background:#0b1220;color:#dbeafe;border-radius:8px;")
        self.progress = QProgressBar()
        self.info_label = QLabel("当前帧：0 | 推理耗时：0 ms | 当前目标：0")
        lay.addWidget(self.viewer, stretch=1)
        lay.addWidget(self.progress)
        lay.addWidget(self.info_label)
        return box

    def _right_panel(self):
        box = QWidget()
        box.setFixedWidth(330)
        lay = QVBoxLayout(box)
        card, stats_lay = self._card("实时统计")
        self.total_label = QLabel("总目标：0")
        self.current_label = QLabel("当前画面：0")
        self.size_label = QLabel("目标尺寸：- / -")
        self.class_label = QLabel("类别统计：暂无")
        for w in (self.total_label, self.current_label, self.size_label, self.class_label):
            stats_lay.addWidget(w)
        lay.addWidget(card)
        log_card, log_lay = self._card("运行日志")
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        clear = QPushButton("清空日志")
        save = QPushButton("保存日志")
        clear.clicked.connect(self.log_text.clear)
        save.clicked.connect(self.save_log)
        log_lay.addWidget(self.log_text, stretch=1)
        log_lay.addLayout(self._row("", clear, save))
        lay.addWidget(log_card, stretch=1)
        return box

    def _row(self, label, *widgets):
        row = QHBoxLayout()
        if label:
            lab = QLabel(label)
            lab.setMinimumWidth(70)
            row.addWidget(lab)
        for w in widgets:
            row.addWidget(w, stretch=1)
        return row

    def _slider(self, mn, mx, val):
        s = QSlider(Qt.Horizontal)
        s.setRange(mn, mx)
        s.setValue(val)
        return s

    def scan_weights(self):
        self.model_combo.clear()
        candidates = []
        for folder in [Path("weights"), Path("runs"), Path(".")]:
            if folder.exists():
                candidates.extend(folder.rglob("*.pt"))
        for p in sorted(set(candidates)):
            self.model_combo.addItem(str(p))
        if self.model_combo.count():
            self.model_path.setText(self.model_combo.currentText())

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "PyTorch weights (*.pt)")
        if path:
            self.model_path.setText(path)

    def browse_source(self):
        mode = self.mode_combo.currentText()
        if mode == "单张图片":
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        elif mode == "图片文件夹":
            path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        elif mode == "本地视频文件":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)")
        else:
            path = ""
        if path:
            self.source_path.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)

    def load_model(self):
        try:
            self.set_state("running", "加载模型")
            QApplication.processEvents()
            self.inferencer.load(self.model_path.text(), self.device_combo.currentText())
            self.set_state("idle", "模型已加载")
            self.log(f"模型加载成功：{self.model_path.text()}", "ok")
        except Exception as exc:
            self.set_state("error", "加载失败")
            self.log(str(exc), "error")
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def build_config(self):
        mode = self.mode_combo.currentText()
        source = self.camera_input.text().strip() if mode == "摄像头/RTSP实时流" else self.source_path.text().strip()
        return {
            "mode": mode,
            "source": source,
            "conf": self.conf_slider.value() / 100.0,
            "iou": self.iou_slider.value() / 100.0,
            "imgsz": int(self.imgsz_combo.currentText()),
            "device": self.device_combo.currentText(),
            "half": self.fp16.isChecked(),
            "line_width": self.line_slider.value(),
            "font_size": self.font_slider.value(),
            "show_conf": self.show_conf.isChecked(),
            "show_id": self.show_id.isChecked(),
            "grid_crop": self.grid_crop.isChecked(),
            "show_grid": self.show_grid.isChecked(),
            "tracker": self.tracker.isChecked(),
            "frame_skip": self.frame_skip.value(),
            "save_image": self.save_image.isChecked(),
            "save_crops": self.save_crops.isChecked(),
            "save_json": self.save_json.isChecked(),
            "save_csv": self.save_csv.isChecked(),
            "save_video": self.save_video.isChecked(),
            "max_items": self.max_items.value(),
            "exporter": __import__("yolo_water_detect.utils.export_tool", fromlist=["ExportManager"]).ExportManager(self.output_dir.text()),
        }

    def start_detection(self):
        if self.inferencer.model is None:
            self.load_model()
            if self.inferencer.model is None:
                return
        try:
            config = self.build_config()
            if not config["source"]:
                QMessageBox.warning(self, "缺少数据源", "请选择图片、文件夹、视频，或填写摄像头/RTSP地址。")
                return
            self.worker = DetectionWorker(self.inferencer, config)
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.progress.connect(lambda a, b: self.progress.setValue(int(a * 100 / b)) if b else None)
            self.worker.log.connect(self.log)
            self.worker.stats.connect(self.update_stats)
            self.worker.failed.connect(lambda msg: (self.set_state("error", "报错"), self.log(msg, "error")))
            self.worker.finished.connect(lambda: self.set_state("idle", "空闲"))
            self.set_state("running", "推理中")
            self.worker.start()
        except Exception as exc:
            self.set_state("error", "启动失败")
            QMessageBox.critical(self, "启动失败", str(exc))

    def pause_detection(self):
        if not self.worker:
            return
        if self.worker._paused:
            self.worker.resume()
            self.pause_btn.setText("暂停")
            self.log("继续检测。", "info")
        else:
            self.worker.pause()
            self.pause_btn.setText("继续")
            self.log("检测已暂停。", "warn")

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
            self.log("正在终止任务...", "warn")

    def update_frame(self, frame, detections, elapsed, idx, total):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.last_pixmap = QPixmap.fromImage(img)
        self.viewer.setPixmap(self.last_pixmap.scaled(self.viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.time_label.value_label.setText(f"{elapsed:.0f}")
        self.info_label.setText(f"当前帧：{idx}/{total if total else '-'} | 推理耗时：{elapsed:.1f} ms | 当前目标：{len(detections)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.last_pixmap:
            self.viewer.setPixmap(self.last_pixmap.scaled(self.viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_stats(self, stats):
        self.total_label.setText(f"总目标：{stats['total_count']}")
        self.current_label.setText(f"当前画面：{stats['current_count']}")
        self.size_label.setText(f"目标尺寸：最大 {stats['max_size']} / 最小 {stats['min_size']}")
        self.class_label.setText("类别统计：" + (", ".join([f"{k}:{v}" for k, v in stats["class_counts"].items()]) or "暂无"))
        self.fps_label.value_label.setText(str(stats.get("avg_fps", 0)))

    def set_state(self, state, text):
        mapping = {"idle": "StatusDotIdle", "running": "StatusDotRunning", "error": "StatusDotError"}
        self.status_dot.setObjectName(mapping[state])
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)
        self.status_text.setText(text)
        self.statusBar().showMessage(f"数据源：{self.source_path.text() or self.camera_input.text()} | 模型：{self.model_path.text()} | 设备：{self.device_combo.currentText()}")

    def log(self, message, level="info"):
        color = {"info": "#e5e7eb", "ok": "#86efac", "warn": "#fde68a", "error": "#fca5a5"}.get(level, "#e5e7eb")
        self.log_text.append(f"<span style='color:{color}'>[{time.strftime('%H:%M:%S')}] {message}</span>")

    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存日志", "detect_log.txt", "Text (*.txt)")
        if path:
            Path(path).write_text(self.log_text.toPlainText(), encoding="utf-8")

    def open_output_dir(self):
        path = Path(self.output_dir.text()).resolve()
        path.mkdir(parents=True, exist_ok=True)
        import os
        os.startfile(str(path))
