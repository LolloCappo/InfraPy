# -*- coding: utf-8 -*-
"""
InfraPy GUI – analysis mixin
------------------------------
AnalysisMixin adds all frequency-domain and time-domain analysis viewers to
IRViewerPG via multiple inheritance.  Every method receives the full window
state through ``self``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
)

try:
    from .style import ROI_COLORS
except ImportError:
    from style import ROI_COLORS  # type: ignore


class AnalysisMixin:
    """Mixin that provides time-domain and frequency-domain analysis windows."""

    # ------------------------------------------------------------------
    # Entry points called from the menu
    # ------------------------------------------------------------------

    def run_time_analysis(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        if self.fs is None:
            print("Sampling frequency not set.")
            return
        print(f"Running Time-Domain analysis with fs={self.fs}")
        self._show_time_analysis_viewer(self.loaded_data, self.fs)

    def run_fft_analysis(self) -> None:
        if self.loaded_data is None or self.fs is None:
            print("Video or sampling frequency missing.")
            return

        data = self.loaded_data
        fs = self.fs
        n_frames, h, w = data.shape

        segment_choice, ok = QInputDialog.getItem(
            self,
            "FFT Segmentation",
            "Do you want to segment the time history?",
            ["No", "Yes"],
            0,
            False,
        )
        if not ok:
            return

        do_segment = segment_choice == "Yes"
        n_segments = 1
        overlap_percent = 0.0

        if do_segment:
            n_segments, ok = QInputDialog.getInt(
                self, "Number of Segments", "Enter number of segments:", 4, 1, 100, 1
            )
            if not ok:
                return

            overlap_percent, ok = QInputDialog.getDouble(
                self,
                "Overlap Percentage",
                "Enter overlap percentage (0-90):",
                50.0,
                0.0,
                90.0,
                1,
            )
            if not ok:
                return

        window_choice, ok = QInputDialog.getItem(
            self, "Windowing", "Apply Hann window?", ["No", "Yes"], 1, False
        )
        if not ok:
            return
        use_hann = window_choice == "Yes"

        if do_segment:
            seg_len = n_frames // n_segments
            if seg_len < 2:
                print("Too many segments for the available frames.")
                return
            step = int(seg_len * (1 - overlap_percent / 100))
            if step <= 0:
                step = 1
            starts = list(range(0, n_frames - seg_len + 1, step))
        else:
            seg_len = n_frames
            starts = [0]

        n_freqs = seg_len // 2 + 1
        fft_freqs = np.fft.rfftfreq(seg_len, d=1 / fs)

        if use_hann:
            window = np.hanning(seg_len)
            U = np.sum(window ** 2) / seg_len
            scale = 2.0 / (seg_len * U)
        else:
            window = np.ones(seg_len)
            scale = 2.0 / seg_len

        progress = QProgressDialog("Computing FFT...", "Cancel", 0, h, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        fft_mag = np.zeros((n_freqs, h, w), dtype=np.float32)
        fft_phase = np.zeros((n_freqs, h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                signal = data[:, i, j]
                segment_mags = []
                segment_phases = []

                for start in starts:
                    seg = signal[start : start + seg_len]
                    if len(seg) < seg_len:
                        continue
                    seg = seg - seg.mean()
                    seg = seg * window
                    fft_result = np.fft.rfft(seg)
                    segment_mags.append(np.abs(fft_result) * scale)
                    segment_phases.append(np.angle(fft_result))

                if segment_mags:
                    fft_mag[:, i, j] = np.mean(segment_mags, axis=0)
                    fft_phase[:, i, j] = np.mean(segment_phases, axis=0)

            progress.setValue(i + 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                print("FFT cancelled.")
                return

        progress.close()

        def spectrum_callback(mean_signal: np.ndarray):
            if do_segment:
                segment_mags = []
                for start in starts:
                    seg = mean_signal[start : start + seg_len]
                    if len(seg) < seg_len:
                        continue
                    seg = seg - seg.mean()
                    seg = seg * window
                    fft_result = np.fft.rfft(seg)
                    segment_mags.append(np.abs(fft_result) * scale)
                return fft_freqs, np.mean(segment_mags, axis=0)
            else:
                sig = mean_signal - mean_signal.mean()
                sig = sig * window
                return fft_freqs, np.abs(np.fft.rfft(sig)) * scale

        self._show_frequency_analysis_viewer(
            data, fft_freqs, fft_mag, fft_phase, spectrum_callback, enable_save_mag=True
        )

    def run_correlation_analysis(self) -> None:
        if self.loaded_data is None:
            print("No video loaded.")
            return
        if self.fs is None:
            print("Sampling frequency not set.")
            return

        data = self.loaded_data
        fs = self.fs
        n_frames, h, w = data.shape
        N = n_frames

        freq_vector = np.linspace(0, fs / 2, 200)
        n_freqs = len(freq_vector)
        t = np.arange(N) / fs

        mag_data = np.empty((n_freqs, h, w), dtype=np.float32)
        phase_data = np.empty((n_freqs, h, w), dtype=np.float32)

        progress = QProgressDialog(
            "Computing Lock-in Correlation...", "Cancel", 0, n_freqs, self
        )
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        for k, fl in enumerate(freq_vector):
            sine = np.sin(2 * np.pi * fl * t)
            cosine = np.cos(2 * np.pi * fl * t)
            sine_3d = sine[:, None, None]
            cosine_3d = cosine[:, None, None]

            X = (2 / N) * np.sum(data * cosine_3d, axis=0)
            Y = (2 / N) * np.sum(data * sine_3d, axis=0)
            mag_data[k] = np.sqrt(X ** 2 + Y ** 2)
            phase_data[k] = np.degrees(np.arctan2(Y, X))

            progress.setValue(k + 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                print("Lock-in correlation cancelled.")
                return

        progress.close()

        def spectrum_callback(mean_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            spectrum_list = []
            for fl in freq_vector:
                sine = np.sin(2 * np.pi * fl * t)
                cosine = np.cos(2 * np.pi * fl * t)
                X_roi = (2 / N) * np.sum(mean_signal * cosine)
                Y_roi = (2 / N) * np.sum(mean_signal * sine)
                spectrum_list.append(np.sqrt(X_roi ** 2 + Y_roi ** 2))
            return freq_vector, np.array(spectrum_list)

        self._show_frequency_analysis_viewer(
            data, freq_vector, mag_data, phase_data, spectrum_callback, enable_save_mag=False
        )

    # ------------------------------------------------------------------
    # Viewer dialogs
    # ------------------------------------------------------------------

    def _show_frequency_analysis_viewer(
        self,
        original_data: np.ndarray,
        freq_vector: np.ndarray,
        mag_data: np.ndarray,
        phase_data: np.ndarray,
        spectrum_callback: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        enable_save_mag: bool = False,
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Frequency Domain Analysis")
        dlg.resize(self.size())
        vbox = QVBoxLayout(dlg)

        top = QHBoxLayout()
        vbox.addLayout(top)

        raw_view = pg.ImageView()
        raw_view.setImage(original_data, autoLevels=True)
        raw_view.ui.menuBtn.hide()
        top.addWidget(raw_view)

        mag_view = pg.ImageView()
        mag_view.setImage(mag_data, autoLevels=True)
        mag_view.ui.menuBtn.hide()
        mag_view.ui.roiBtn.hide()
        top.addWidget(mag_view)

        ph_view = pg.ImageView()
        ph_view.setImage(phase_data, autoLevels=True)
        ph_view.ui.menuBtn.hide()
        ph_view.ui.roiBtn.hide()
        top.addWidget(ph_view)

        spectrum_plot = pg.PlotWidget()
        spectrum_plot.setLabel("bottom", "Frequency [Hz]")
        spectrum_plot.setLabel("left", "Amplitude")
        vbox.addWidget(spectrum_plot)

        spectrum_curve = spectrum_plot.plot([], pen="y")
        cursor_line = pg.InfiniteLine(angle=90, movable=True, pen="r")
        spectrum_plot.addItem(cursor_line)

        roi = pg.RectROI([30, 30], [40, 40], pen=pg.mkPen("r", width=2))
        raw_view.addItem(roi)

        raw_view.setMinimumSize(400, 400)
        mag_view.setMinimumSize(400, 400)
        ph_view.setMinimumSize(400, 400)
        spectrum_plot.setMinimumHeight(150)

        if enable_save_mag:
            save_row = QHBoxLayout()
            vbox.addLayout(save_row)

            save_btn = QPushButton("Save Mag/Phase as separate .npy")
            save_btn.setToolTip("Choose a base name once; saves ..._mag.npy, ..._phase.npy")
            save_row.addWidget(save_btn)
            save_row.addStretch(1)

            def _save_many() -> None:
                path, _ = QFileDialog.getSaveFileName(
                    dlg, "Save FFT outputs (base name)", "fft.npy", "NumPy binary (*.npy)"
                )
                if not path:
                    return
                base = Path(path)
                if base.suffix.lower() == ".npy":
                    base = base.with_suffix("")
                mag_path = base.with_name(base.name + "_mag.npy")
                phase_path = base.with_name(base.name + "_phase.npy")
                freqs_path = base.with_name(base.name + "_freqs.npy")
                try:
                    np.save(str(mag_path), mag_data.T)
                    np.save(str(phase_path), phase_data.T)
                    np.save(str(freqs_path), freq_vector)
                    print(f"Saved: {mag_path}")
                    print(f"Saved: {phase_path}")
                    print(f"Saved: {freqs_path}")
                except Exception as e:
                    print(f"Failed to save FFT outputs: {e}")

            save_btn.clicked.connect(_save_many)

        def update_fft_images_from_cursor() -> None:
            freq_pos = cursor_line.value()
            idx = np.abs(freq_vector - freq_pos).argmin()
            self.current_fft_index = int(idx)
            mag_view.setImage(mag_data[idx][np.newaxis], autoLevels=False)
            ph_view.setImage(phase_data[idx][np.newaxis], autoLevels=False)

        def update_spectrum() -> None:
            try:
                h_, w_ = roi.size()
                pos = roi.pos()
                y0, x0 = math.floor(pos[0]), math.floor(pos[1])
                y1, x1 = math.ceil(pos[0] + w_), math.ceil(pos[1] + h_)

                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(original_data.shape[2], x1)
                y1 = min(original_data.shape[1], y1)

                if x1 <= x0 or y1 <= y0:
                    print("ROI outside bounds or zero area.")
                    return

                roi_data = original_data[:, y0:y1, x0:x1]
                mean_signal = roi_data.mean(axis=(1, 2))
                freqs, spectrum = spectrum_callback(mean_signal)
                spectrum_curve.setData(freqs, spectrum)
                spectrum_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                spectrum_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)

                if self.current_fft_index < len(freqs):
                    cursor_line.setPos(freqs[self.current_fft_index])
                update_fft_images_from_cursor()

            except Exception as e:
                print("Spectrum update failed:", e)

        roi.sigRegionChanged.connect(update_spectrum)
        cursor_line.sigPositionChanged.connect(update_fft_images_from_cursor)
        update_spectrum()
        dlg.exec_()

    def _show_time_analysis_viewer(self, data: np.ndarray, fs: float) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Time Domain Analysis")
        dlg.resize(self.size())
        layout = QVBoxLayout(dlg)

        top = QHBoxLayout()
        layout.addLayout(top)

        raw_view = pg.ImageView()
        raw_view.setImage(data, autoLevels=True)
        raw_view.ui.menuBtn.hide()
        raw_view.ui.roiBtn.hide()
        top.addWidget(raw_view)

        time_plot = pg.PlotWidget()
        time_plot.setLabel("bottom", "Time [s]")
        time_plot.setLabel("left", "Intensity")
        layout.addWidget(time_plot)

        roi_list: list = []
        curves: list = []

        ctrl_layout = QHBoxLayout()
        layout.addLayout(ctrl_layout)

        detrend_cb = QPushButton("Detrend")
        detrend_cb.setCheckable(True)
        filter_cb = QPushButton("Bandpass")
        filter_cb.setCheckable(True)
        smooth_cb = QPushButton("Smooth")
        smooth_cb.setCheckable(True)

        ctrl_layout.addWidget(detrend_cb)
        ctrl_layout.addWidget(filter_cb)
        ctrl_layout.addWidget(smooth_cb)

        t = np.arange(data.shape[0]) / fs

        def process_signal(sig: np.ndarray) -> np.ndarray:
            out = sig.copy()
            if detrend_cb.isChecked():
                out = out - np.polyval(np.polyfit(t, out, 1), t)
            if filter_cb.isChecked():
                from scipy.signal import butter, filtfilt
                b, a = butter(3, [0.1, 0.3], btype="band")
                out = filtfilt(b, a, out)
            if smooth_cb.isChecked():
                win = 5
                out = np.convolve(out, np.ones(win) / win, mode="same")
            return out

        def update_all_traces() -> None:
            for roi, curve in zip(roi_list, curves):
                pos = roi.pos()
                size = roi.size()
                x0, y0 = int(pos[0]), int(pos[1])
                w_, h_ = int(size[0]), int(size[1])
                x1, y1 = x0 + w_, y0 + h_
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(data.shape[2], x1)
                y1 = min(data.shape[1], y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                roi_data = data[:, y0:y1, x0:x1]
                mean_signal = roi_data.mean(axis=(1, 2))
                curve.setData(t, process_signal(mean_signal))

        def add_roi() -> None:
            color = ROI_COLORS[len(roi_list) % len(ROI_COLORS)]
            roi = pg.RectROI(
                [30 + len(roi_list) * 10, 30], [40, 40], pen=pg.mkPen(color, width=2)
            )
            raw_view.addItem(roi)
            roi_list.append(roi)
            curve = time_plot.plot(pen=pg.mkPen(color, width=2))
            curves.append(curve)
            roi.sigRegionChanged.connect(update_all_traces)
            update_all_traces()

        add_roi()

        from PyQt5.QtCore import Qt as _Qt
        raw_view.scene.sigMouseClicked.connect(
            lambda ev: add_roi() if ev.modifiers() & _Qt.ControlModifier else None
        )

        detrend_cb.clicked.connect(update_all_traces)
        filter_cb.clicked.connect(update_all_traces)
        smooth_cb.clicked.connect(update_all_traces)

        dlg.exec_()
