from __future__ import annotations

import time

import cv2
import numpy as np

from ultralytics import YOLO

from .crop_grid import draw_grid, split_grid


class YoloInferencer:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.names = {}

    def load(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.names = getattr(self.model.model, "names", {}) or getattr(self.model, "names", {}) or {}
        return self

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device="cpu",
        half=False,
        line_width=2,
        font_scale=0.6,
        show_conf=True,
        grid=False,
        show_grid=False,
        tracker=False,
    ):
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        start = time.perf_counter()
        if grid:
            annotated, detections = self._infer_grid(
                frame_bgr, conf, iou, imgsz, device, half, line_width, font_scale, show_conf
            )
        else:
            result = self._run_model(frame_bgr, conf, iou, imgsz, device, half, tracker)
            detections = self._extract_detections(result)
            annotated = result.plot(line_width=line_width, font_size=max(8, int(font_scale * 18)))
        if show_grid:
            annotated = draw_grid(annotated)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return annotated, detections, elapsed_ms

    def _run_model(self, frame_bgr, conf, iou, imgsz, device, half, tracker=False):
        kwargs = dict(source=frame_bgr, conf=conf, iou=iou, imgsz=imgsz, device=device, half=half, verbose=False)
        if tracker:
            results = self.model.track(persist=True, tracker="bytetrack.yaml", **kwargs)
        else:
            results = self.model.predict(**kwargs)
        return results[0]

    def _infer_grid(self, frame_bgr, conf, iou, imgsz, device, half, line_width, font_scale, show_conf):
        detections = []
        annotated = frame_bgr.copy()
        for tile in split_grid(frame_bgr):
            result = self._run_model(tile.image, conf, iou, imgsz, device, half, False)
            for det in self._extract_detections(result):
                x1, y1, x2, y2 = det["xyxy"]
                det["xyxy"] = [x1 + tile.x1, y1 + tile.y1, x2 + tile.x1, y2 + tile.y1]
                detections.append(det)
        self._draw_detections(annotated, detections, line_width, font_scale, show_conf)
        return annotated, detections

    def _extract_detections(self, result) -> list[dict]:
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        names = getattr(result, "names", None) or self.names or {}
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
            conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            track_id = None
            if getattr(boxes, "id", None) is not None and boxes.id is not None:
                track_id = int(boxes.id[i].detach().cpu().item())
            detections.append(
                {
                    "xyxy": xyxy,
                    "class_id": cls_id,
                    "class_name": str(names.get(cls_id, cls_id)),
                    "confidence": conf,
                    "track_id": track_id,
                }
            )
        return detections

    @staticmethod
    def _draw_detections(image, detections, line_width=2, font_scale=0.6, show_conf=True):
        palette = [(0, 102, 204), (0, 170, 90), (245, 158, 11), (239, 68, 68), (147, 51, 234)]
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            color = palette[det.get("class_id", 0) % len(palette)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
            label = det.get("class_name", "object")
            if show_conf:
                label += f" {det.get('confidence', 0):.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, line_width),
                cv2.LINE_AA,
            )
