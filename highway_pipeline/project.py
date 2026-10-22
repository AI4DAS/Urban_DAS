"""Configuration-driven Highway vehicle-trace business workflow."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate

from . import das_prep
from .trace_associate import associate_image


class HighwayProject:
    def __init__(self, config_file):
        self.config_file = Path(config_file).resolve()
        self.src_dir = self.config_file.parent.parent
        self.config = json.loads(self.config_file.read_text(encoding="utf-8"))

    def _path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.src_dir / path).resolve()

    def _trace(self, trace_name):
        try:
            return self.config["traces"][str(trace_name)]
        except KeyError as exc:
            choices = ", ".join(self.config["traces"])
            raise KeyError(f"Unknown trace {trace_name!r}; choose one of: {choices}") from exc

    @staticmethod
    def _reset_directory(path):
        path = Path(path).resolve()
        if len(path.parts) < 5:
            raise ValueError(f"Refusing to reset broad path: {path}")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    @staticmethod
    def _update_json(path, key, value):
        document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        document[key] = value
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _deleted_channels(delete_ranges):
        return [channel for start, end in delete_ranges for channel in range(start, end)]

    def _prepare_image(self, h5_file, trace):
        ch1, ch2 = trace["channels"]
        settings = self.config["preprocess"]
        data = das_prep.read_das(h5_file, ch1=ch1, ch2=ch2)
        data = das_prep.das_preprocess(data)
        for factor in settings["decimation"]:
            data = decimate(data, factor, axis=-1).astype("float32")

        delete_ranges = trace["delete"]
        if delete_ranges:
            data = np.delete(data, self._deleted_channels(delete_ranges), axis=0)

        sample_interval = self.config["data"]["dt"] * np.prod(settings["decimation"])
        data = das_prep.tapering(data, settings["taper_alpha"])
        data = das_prep.lowpass(data, sample_interval, settings["lowpass_hz"])
        data = das_prep.das_preprocess(data)
        data = data / data.std(axis=1, keepdims=True)
        metadata = {
            "h5_file_path": str(h5_file),
            "M": settings["decimation"],
            "delet": delete_ranges,
            "ch1": ch1,
            "ch2": ch2,
            "nch": len(data),
            "nt": len(data[0]),
            "Mdt": float(sample_interval),
            "dx": self.config["data"]["dx"],
        }
        return data, metadata

    @staticmethod
    def _direction_data(data):
        spectrum = np.fft.fftshift(np.fft.fft2(data))
        time_frequency = np.fft.fftshift(np.fft.fftfreq(data.shape[1], 1.0))
        space_frequency = np.fft.fftshift(np.fft.fftfreq(data.shape[0], 1.0))
        sign = time_frequency[None, :] / (space_frequency[:, None] + 1e-10)
        positive = spectrum.copy()
        positive[sign <= 0] = 0
        negative = spectrum.copy()
        negative[sign >= 0] = 0
        return (
            np.fft.ifft2(np.fft.ifftshift(positive)).real,
            np.fft.ifft2(np.fft.ifftshift(negative)).real,
        )

    @staticmethod
    def _save_image(data, path, figure_size, clim_rate):
        figure = plt.figure(figsize=figure_size)
        limit = data.std() * clim_rate
        plt.imshow(data.T, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit)
        plt.axis("off")
        plt.xticks([])
        figure.savefig(path, bbox_inches="tight", pad_inches=-0.1)
        plt.close(figure)

    def prepare(self, trace_name):
        """Generate org/pos/neg images and metadata JSON from H5 files."""
        trace = self._trace(trace_name)
        output_dir = self._path(trace["output"])
        image_dir = output_dir / "pic"
        self._reset_directory(image_dir)

        data_settings = self.config["data"]
        h5_files = sorted(self._path(data_settings["directory"]).glob(data_settings["pattern"]))
        start, end = data_settings.get("file_slice", [0, None])
        h5_files = h5_files[start:end]
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {self._path(data_settings['directory'])}")

        for index, h5_file in enumerate(h5_files):
            data, metadata = self._prepare_image(h5_file, trace)
            positive, negative = self._direction_data(data)
            for suffix, image_data in (("org", data), ("pos", positive), ("neg", negative)):
                self._save_image(
                    image_data,
                    image_dir / f"{index}_{suffix}.png",
                    trace["figure_size"],
                    trace["clim_rate"],
                )
            self._update_json(image_dir / f"{index}.json", "h5 matadata", metadata)
        return len(h5_files)

    def _images_to_patches(self, image_dir, patch_dir):
        settings = self.config["preprocess"]
        patch_size = settings["patch_size"]
        padding = settings["padding"]
        value = settings["padding_value"]
        dimensions = {}
        for image_path in sorted(image_dir.glob("*.png")):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            height, width = image.shape[:2]
            dimensions[image_path.stem] = (height, width)
            columns = width // patch_size
            for row in range(height // patch_size):
                for column in range(columns):
                    patch = image[
                        row * patch_size : (row + 1) * patch_size,
                        column * patch_size : (column + 1) * patch_size,
                    ]
                    patch = cv2.copyMakeBorder(
                        patch, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=value
                    )
                    patch_id = row * columns + column
                    cv2.imwrite(str(patch_dir / f"{image_path.stem}_{patch_id:05d}.png"), patch)
        if not dimensions:
            raise FileNotFoundError(f"No PNG images found in {image_dir}; run prepare first")
        return dimensions

    def _run_yolo(self, patch_dir, yolo_dir):
        model = self.config["model"]
        repository = self._path(model["repository"])
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        yolo = importlib.import_module("detect")
        yolo.run(
            weights=str(self._path(model["weight"])),
            source=str(patch_dir),
            project=str(yolo_dir),
            name="yolo",
            exist_ok=True,
            iou_thres=model["iou_threshold"],
            conf_thres=model["confidence_threshold"],
            imgsz=tuple(model["image_size"]),
            save_txt=True,
            nosave=True,
        )

    @staticmethod
    def _file_start_time(filename):
        timestamp = " ".join(Path(filename).stem.split("_")[-2:])
        return np.datetime64(datetime.strptime(timestamp, "%Y%m%d %H%M%S.%f"))

    def _pixel_to_time(self, json_path):
        document = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = document["h5 matadata"]
        start_time = self._file_start_time(metadata["h5_file_path"])
        for option in ("neg", "pos", "org"):
            trace = document[option]
            image_height, image_width = trace["img"]
            physical = []
            for vehicle in trace["associate pixel"]:
                vehicle_segments = []
                for point1, point2, direction in vehicle:
                    converted = []
                    for x, y in (point1, point2):
                        channel = int(x / image_width * metadata["nch"])
                        time_index = int(y / image_height * metadata["nt"])
                        distance = channel * metadata["dx"]
                        seconds = int(time_index * metadata["Mdt"])
                        timestamp = np.datetime_as_string(start_time + np.timedelta64(seconds, "s"))
                        converted.append([channel, time_index, distance, timestamp])
                    converted.append(direction)
                    vehicle_segments.append(converted)
                physical.append(vehicle_segments)
            trace["trace_real_dic"] = physical
        json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _draw_traces(image_path, json_path):
        image = cv2.imread(str(image_path))
        document = json.loads(json_path.read_text(encoding="utf-8"))
        option = image_path.stem.rsplit("_", 1)[1]
        for vehicle in document[option]["associate pixel"]:
            points = []
            for point1, point2, direction in vehicle:
                if option == "org" or (option == "neg" and direction == 1) or (option == "pos" and direction == 0):
                    points.extend((point1, point2))
            if points:
                cv2.polylines(image, np.int32([np.asarray(points)]), False, (0, 255, 0), 3)
        cv2.imwrite(str(image_path), image)

    def trace(self, trace_name):
        """Run patch creation, YOLO inference, association and visualization."""
        trace = self._trace(trace_name)
        output_dir = self._path(trace["output"])
        image_dir = output_dir / "pic"
        patch_dir = output_dir / "patches"
        yolo_dir = output_dir / "yolo"
        visual_dir = output_dir / "predict_vis_result"
        for directory in (patch_dir, yolo_dir, visual_dir):
            self._reset_directory(directory)

        dimensions = self._images_to_patches(image_dir, patch_dir)
        self._run_yolo(patch_dir, yolo_dir)
        labels_dir = yolo_dir / "yolo" / "labels"
        settings = self.config["preprocess"]
        summaries = []
        for image_name, (height, width) in dimensions.items():
            json_path = image_dir / f"{image_name.rsplit('_', 1)[0]}.json"
            option = image_name.rsplit("_", 1)[1]
            payload = associate_image(
                image_name,
                labels_dir,
                height,
                width,
                patch_size=settings["patch_size"],
                padding=settings["padding"],
                threshold=trace["associate_threshold"],
                minimum_segments=trace["minimum_segments"],
            )
            self._update_json(json_path, option, payload)
            summaries.append(f"{image_name}: {payload['associate performance']}")

        for json_path in sorted(image_dir.glob("*.json")):
            self._pixel_to_time(json_path)
        for image_path in sorted(image_dir.glob("*.png")):
            target = visual_dir / image_path.name
            shutil.copy2(image_path, target)
            self._draw_traces(target, image_dir / f"{image_path.stem.rsplit('_', 1)[0]}.json")
        return summaries

    def run(self, trace_name):
        count = self.prepare(trace_name)
        summaries = self.trace(trace_name)
        return count, summaries
