from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import cv2


class ExportManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.crops_dir = self.output_dir / "crops"
        self.json_dir = self.output_dir / "json"
        self.video_dir = self.output_dir / "videos"
        for p in (self.images_dir, self.crops_dir, self.json_dir, self.video_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "detections.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    [
                        "source",
                        "frame",
                        "class_id",
                        "class_name",
                        "confidence",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "width",
                        "height",
                    ]
                )

    def save_image(self, image, stem: str):
        path = self.images_dir / f"{stem}.jpg"
        cv2.imwrite(str(path), image)
        return path

    def save_crops(self, frame, detections, stem: str):
        saved = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                continue
            cls_name = str(det.get("class_name", "object")).replace(" ", "_")
            path = self.crops_dir / f"{stem}_{i:04d}_{cls_name}.jpg"
            cv2.imwrite(str(path), crop)
            saved.append(path)
        return saved

    def save_json(self, metadata: Mapping, stem: str):
        path = self.json_dir / f"{stem}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return path

    def append_csv(self, source: str, frame_index: int, detections):
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for det in detections:
                x1, y1, x2, y2 = det["xyxy"]
                writer.writerow(
                    [
                        source,
                        frame_index,
                        det.get("class_id", ""),
                        det.get("class_name", ""),
                        f"{det.get('confidence', 0):.4f}",
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        int(x2 - x1),
                        int(y2 - y1),
                    ]
                )
