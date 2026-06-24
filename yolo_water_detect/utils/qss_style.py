APP_QSS = """
QMainWindow, QWidget {
    background: #f3f8fd;
    color: #1f2937;
    font-family: "Microsoft YaHei", "Segoe UI", Arial;
    font-size: 13px;
}
QFrame#Header { background: #ffffff; border-bottom: 1px solid #cfe0ef; }
QLabel#TitleLabel { color: #003b73; font-size: 22px; font-weight: 700; }
QLabel#MetricValue { color: #005cb8; font-size: 18px; font-weight: 700; }
QLabel#MetricName { color: #64748b; font-size: 11px; }
QLabel#StatusDotIdle, QLabel#StatusDotRunning, QLabel#StatusDotError {
    min-width: 14px; max-width: 14px; min-height: 14px; max-height: 14px; border-radius: 7px;
}
QLabel#StatusDotIdle { background: #22c55e; }
QLabel#StatusDotRunning { background: #0ea5e9; }
QLabel#StatusDotError { background: #ef4444; }
QFrame#Card { background: #ffffff; border: 1px solid #d7e6f3; border-radius: 8px; }
QLabel#CardTitle { color: #003b73; font-size: 14px; font-weight: 700; }
QPushButton {
    background: #e8f3ff; border: 1px solid #9bc5ed; border-radius: 6px;
    padding: 7px 10px; color: #064b85; font-weight: 600;
}
QPushButton:hover { background: #d7ecff; border-color: #5aa2df; }
QPushButton:pressed { background: #beddf7; }
QPushButton#PrimaryButton { background: #0066cc; color: #ffffff; border: 1px solid #0057ad; }
QPushButton#PrimaryButton:hover { background: #0b75df; }
QPushButton#DangerButton { background: #ef4444; color: #ffffff; border: 1px solid #dc2626; }
QPushButton#WarnButton { background: #f59e0b; color: #ffffff; border: 1px solid #d97706; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #f8fbff; border: 1px solid #b9d3ea; border-radius: 5px; padding: 5px;
}
QTextEdit { background: #0f172a; color: #e5e7eb; border: 1px solid #1e293b; border-radius: 8px; padding: 8px; }
QProgressBar { border: 1px solid #a7c7e7; border-radius: 6px; background: #edf6ff; text-align: center; height: 16px; }
QProgressBar::chunk { background: #0066cc; border-radius: 5px; }
QSlider::groove:horizontal { height: 6px; background: #d8e8f7; border-radius: 3px; }
QSlider::handle:horizontal { background: #0066cc; width: 14px; margin: -5px 0; border-radius: 7px; }
QSplitter::handle { background: #d8e8f7; }
"""
