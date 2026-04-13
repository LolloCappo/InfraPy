# -*- coding: utf-8 -*-
"""
InfraPy GUI – crop mixin
--------------------------
CropMixin adds all spatial-crop functionality to IRViewerPG via multiple
inheritance.  Every method receives the full window state through ``self``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg

try:
    from .widgets import CircleParamsDialog
except ImportError:
    from widgets import CircleParamsDialog  # type: ignore
from PyQt5.QtWidgets import QDialog


class CropMixin:
    """Mixin that provides rectangular and circular ROI cropping."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_image_item(self):
        """Return the underlying ImageItem from the main ImageView."""
        img_item = getattr(self.image_view, "imageItem", None)
        if img_item is None and hasattr(self.image_view, "getImageItem"):
            img_item = self.image_view.getImageItem()
        return img_item

    def _ensure_no_existing_roi(self) -> None:
        """Remove any existing crop ROI from the view."""
        if self.crop_roi is not None:
            try:
                self.image_view.removeItem(self.crop_roi)
            except Exception:
                pass
            self.crop_roi = None
        self.apply_crop_btn.setEnabled(False)
        self.remove_roi_btn.setEnabled(False)

    def _enforce_circular_roi(self) -> None:
        """Keep the EllipseROI strictly circular (1:1), preserving its center."""
        if self.crop_roi is None or self._circle_update_lock:
            return
        try:
            self._circle_update_lock = True
            pos = self.crop_roi.pos()
            size = self.crop_roi.size()
            w = float(size[0])
            h = float(size[1])
            if w <= 0 or h <= 0:
                return
            s = min(w, h)
            cx = float(pos[0]) + w / 2.0
            cy = float(pos[1]) + h / 2.0
            self.crop_roi.setPos([cx - s / 2.0, cy - s / 2.0], finish=False)
            self.crop_roi.setSize([s, s], finish=False)
        finally:
            self._circle_update_lock = False
        self._update_circle_info_label()

    def _update_circle_info_label(self) -> None:
        """Show circle center (x, y) and diameter (px) in the status bar label."""
        if self.crop_roi is None or not self._last_crop_was_ellipse:
            self.circle_info_label.setText("Circle ROI: —")
            return
        bounds = self._roi_bounds_clamped()
        if bounds is None:
            self.circle_info_label.setText("Circle ROI: —")
            return
        x0, y0, x1, y1 = bounds
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        d = min(x1 - x0, y1 - y0)
        self.circle_info_label.setText(
            f"Circle ROI: center=({int(round(cx))}, {int(round(cy))}), d={int(round(d))} px"
        )

    def _roi_bounds_clamped(self) -> Optional[tuple]:
        """
        Return (x0, y0, x1, y1) integer bounds of the current ROI in array
        coordinates, using ROI.getArraySlice for accurate transform-aware mapping.
        """
        if self.loaded_data is None or self.crop_roi is None:
            return None

        img2d = self.loaded_data[self.current_frame]
        img_item = self._get_image_item()
        if img_item is None:
            print("Internal error: ImageItem not found.")
            return None

        try:
            slices, _ = self.crop_roi.getArraySlice(img2d, img_item)
            sy, sx = slices

            y0 = 0 if sy.start is None else int(sy.start)
            y1 = img2d.shape[0] if sy.stop is None else int(sy.stop)
            x0 = 0 if sx.start is None else int(sx.start)
            x1 = img2d.shape[1] if sx.stop is None else int(sx.stop)

            y0 = max(0, min(y0, img2d.shape[0]))
            y1 = max(0, min(y1, img2d.shape[0]))
            x0 = max(0, min(x0, img2d.shape[1]))
            x1 = max(0, min(x1, img2d.shape[1]))

            if x1 <= x0 or y1 <= y0:
                return None

            return x0, y0, x1, y1
        except Exception as e:
            print(f"Failed to compute ROI slice: {e}")
            return None

    # ------------------------------------------------------------------
    # Public ROI actions
    # ------------------------------------------------------------------

    def add_rect_roi(self) -> None:
        """Add a rectangular ROI to the image view."""
        if self.loaded_data is None:
            print("No video loaded.")
            return
        self._ensure_no_existing_roi()
        _, H, W = self.loaded_data.shape
        w = max(20, int(W * 0.25))
        h = max(20, int(H * 0.25))
        x = int((W - w) / 2)
        y = int((H - h) / 2)
        self.crop_roi = pg.RectROI([x, y], [w, h], pen=pg.mkPen("y", width=2))
        self.image_view.addItem(self.crop_roi)
        self._last_crop_was_ellipse = False
        self.apply_crop_btn.setEnabled(True)
        self.remove_roi_btn.setEnabled(True)

    def add_circle_roi(self) -> None:
        """Add a circular ROI and enforce 1:1 aspect during any resize/move."""
        if self.loaded_data is None:
            print("No video loaded.")
            return
        self._ensure_no_existing_roi()
        _, H, W = self.loaded_data.shape
        d = max(20, int(min(W, H) * 0.4))
        x = int((W - d) / 2)
        y = int((H - d) / 2)
        self.crop_roi = pg.EllipseROI([x, y], [d, d], pen=pg.mkPen("c", width=2))
        self.image_view.addItem(self.crop_roi)
        self.crop_roi.sigRegionChanged.connect(self._enforce_circular_roi)
        self._enforce_circular_roi()
        self._last_crop_was_ellipse = True
        self.apply_crop_btn.setEnabled(True)
        self.remove_roi_btn.setEnabled(True)
        self._update_circle_info_label()

    def set_circle_by_numbers(self) -> None:
        """Open a dialog to type center and diameter, then place a circle ROI."""
        if self.loaded_data is None:
            print("No video loaded.")
            return
        _, H, W = self.loaded_data.shape

        if self.crop_roi is not None and self._last_crop_was_ellipse:
            bounds = self._roi_bounds_clamped()
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                init_cx = int(round((x0 + x1) / 2.0))
                init_cy = int(round((y0 + y1) / 2.0))
                init_d = int(round(min(x1 - x0, y1 - y0)))
            else:
                init_cx, init_cy, init_d = W // 2, H // 2, max(1, min(W, H) // 2)
        else:
            init_cx, init_cy, init_d = W // 2, H // 2, max(1, min(W, H) // 2)

        dlg = CircleParamsDialog(self, W, H, init_cx, init_cy, init_d)
        if dlg.exec_() != QDialog.Accepted:
            return
        cx, cy, d = dlg.values()

        d = max(1, min(d, min(W, H)))
        half = d / 2.0
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        x0 = max(0, min(x0, W - d))
        y0 = max(0, min(y0, H - d))

        if self.crop_roi is None or not self._last_crop_was_ellipse:
            self._ensure_no_existing_roi()
            self.crop_roi = pg.EllipseROI([x0, y0], [d, d], pen=pg.mkPen("c", width=2))
            self.image_view.addItem(self.crop_roi)
            self.crop_roi.sigRegionChanged.connect(self._enforce_circular_roi)
            self._last_crop_was_ellipse = True
            self.remove_roi_btn.setEnabled(True)
        else:
            self.crop_roi.setPos([x0, y0], finish=False)
            self.crop_roi.setSize([d, d], finish=False)

        self._enforce_circular_roi()
        self.apply_crop_btn.setEnabled(True)
        self._update_circle_info_label()

    def remove_crop_roi(self) -> None:
        """Remove the ROI overlay without changing data."""
        self._ensure_no_existing_roi()
        self.circle_info_label.setText("Circle ROI: —")

    def apply_spatial_crop(self) -> None:
        """
        Apply the current ROI to crop the stack (t, y, x).
        Rect ROI → rectangular crop.
        Circle ROI → rectangular crop + zeros outside the circle.
        """
        if self.loaded_data is None or self.crop_roi is None:
            print("No ROI to apply.")
            return

        bounds = self._roi_bounds_clamped()
        if bounds is None:
            print("ROI out of bounds or zero area.")
            return

        x0, y0, x1, y1 = bounds

        hw = (
            self.image_view.getHistogramWidget()
            if hasattr(self.image_view, "getHistogramWidget")
            else self.image_view.ui.histogram
        )
        prev_levels = hw.getLevels()

        self._backup_loaded_data = self.loaded_data
        self._backup_original_data = self.original_data

        cropped_loaded = self.loaded_data[:, y0:y1, x0:x1]
        cropped_original = (
            None
            if self.original_data is None
            else self.original_data[:, y0:y1, x0:x1]
        )

        if self._last_crop_was_ellipse:
            h = y1 - y0
            w = x1 - x0
            r = min(h, w) / 2.0
            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
            yy, xx = np.ogrid[:h, :w]
            circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)

            mask_loaded = circle.astype(cropped_loaded.dtype, copy=False)[None, :, :]
            cropped_loaded = cropped_loaded * mask_loaded

            if cropped_original is not None:
                mask_original = circle.astype(cropped_original.dtype, copy=False)[None, :, :]
                cropped_original = cropped_original * mask_original

        self.loaded_data = cropped_loaded
        if cropped_original is not None:
            self.original_data = cropped_original

        self._ensure_no_existing_roi()
        self.current_frame = 0
        self._init_slider()
        self.update_viewer()
        hw.setLevels(*prev_levels)

        self.undo_crop_btn.setEnabled(True)
        print(f"Cropped to x:[{x0},{x1}), y:[{y0},{y1}) ; shape -> {self.loaded_data.shape}")

    def undo_spatial_crop(self) -> None:
        """Restore dataset to the state before the last crop."""
        if self._backup_loaded_data is None:
            print("No crop to undo.")
            return
        self.loaded_data = self._backup_loaded_data
        self.original_data = self._backup_original_data
        self._backup_loaded_data = None
        self._backup_original_data = None
        self.current_frame = 0
        self._init_slider()
        self.update_viewer()
        self.undo_crop_btn.setEnabled(False)
        print("Crop undone.")
