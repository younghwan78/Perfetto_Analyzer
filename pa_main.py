import sys
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List
from pathlib import Path
import datetime

# --- PyQt5 and Matplotlib Imports ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QTabWidget, QSizePolicy, QProgressBar,
                             QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from PyQt5 import QtGui

# --- Systrace 분석 엔진 import ---
from systrace_engine import SystraceEngine


# --- Stream object to redirect stdout to the UI ---
class Stream(QObject):
    """Redirects console output to a PyQt widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass


class PerfettoAnalyzerApp(QMainWindow):
    """Main application window for the Perfetto Analyzer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perfetto Systrace Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Default paths
        self.config_path = "D:/10_Codes/05_Perfetto_analyzer/event_config.json"
        self.trace_path = "D:/10_Codes/05_Perfetto_analyzer/trace.systrace"

        self.initUI()

        # Redirect stdout to the log window
        sys.stdout = Stream(newText=self.on_new_text)

    def initUI(self):
        """Sets up the user interface."""
        # --- Dark theme & fancy style ---
        dark_stylesheet = """
        QWidget {
            background-color: #232629;
            color: #f0f0f0;
            font-family: 'Segoe UI', 'Malgun Gothic', Arial, sans-serif;
            font-size: 11pt;
        }
        QMainWindow {
            background-color: #232629;
        }
        QLabel {
            color: #f0f0f0;
        }
        QTextEdit, QLineEdit {
            background-color: #181a1b;
            color: #f0f0f0;
            border: 1px solid #444;
            border-radius: 6px;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            background: #181a1b;
        }
        QTabBar::tab {
            background: #232629;
            color: #f0f0f0;
            border: 1px solid #444;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            padding: 8px 18px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #2d3136;
            color: #ffb347;
        }
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #444857, stop:1 #232629);
            color: #f0f0f0;
            border: 1.5px solid #555;
            border-radius: 10px;
            padding: 8px 18px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #5e5e7a, stop:1 #2d3136);
            color: #ffb347;
            border: 2px solid #ffb347;
        }
        QPushButton:pressed {
            background: #181a1b;
            color: #ffb347;
        }
        QProgressBar {
            background-color: #181a1b;
            color: #f0f0f0;
            border: 1px solid #444;
            border-radius: 8px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #ffb347, stop:1 #ffcc80);
            border-radius: 8px;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #232629;
            border: 1px solid #444;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #444857;
            border-radius: 6px;
        }
        """
        self.setStyleSheet(dark_stylesheet)

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Controls) ---
        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout, 1)

        # Config file selection
        btn_config = QPushButton("Select Config JSON")
        btn_config.clicked.connect(self.select_config_file)
        self.lbl_config = QLabel(f"Config: {self.config_path}")
        self.lbl_config.setWordWrap(True)
        
        # Trace file selection
        btn_trace = QPushButton("Select Systrace File")
        btn_trace.clicked.connect(self.select_trace_file)
        self.lbl_trace = QLabel(f"Trace: {self.trace_path}")
        self.lbl_trace.setWordWrap(True)

        # Max lines input
        max_lines_layout = QHBoxLayout()
        self.max_lines_edit = QLineEdit()
        self.max_lines_edit.setPlaceholderText("Max lines (default: 2000000)")
        self.max_lines_edit.setText("2000000")
        max_lines_label = QLabel("Max lines to read:")
        max_lines_layout.addWidget(max_lines_label)
        max_lines_layout.addWidget(self.max_lines_edit)
        controls_layout.addLayout(max_lines_layout)

        # Run button
        btn_run = QPushButton("Run Analysis")
        btn_run.clicked.connect(self.run_analysis)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        # Save Results buttons
        btn_save_csv = QPushButton("Save Results as CSV")
        btn_save_csv.clicked.connect(self.save_results_csv)
        btn_save_sqlite = QPushButton("Save Results as SQLite")
        btn_save_sqlite.clicked.connect(self.save_results_sqlite)
        btn_save_perfetto = QPushButton("Save Results as Perfetto JSON")
        btn_save_perfetto.clicked.connect(self.save_results_perfetto_json)

        # Log window
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        controls_layout.addWidget(btn_config)
        controls_layout.addWidget(self.lbl_config)
        controls_layout.addWidget(btn_trace)
        controls_layout.addWidget(self.lbl_trace)
        controls_layout.addWidget(btn_run)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(btn_save_csv)
        controls_layout.addWidget(btn_save_sqlite)
        controls_layout.addWidget(btn_save_perfetto)
        controls_layout.addWidget(QLabel("Logs:"))
        controls_layout.addWidget(self.log_box)

        # --- Right Panel (Plots) ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 3)
    
    def on_new_text(self, text):
        """Appends text to the log box."""
        self.log_box.moveCursor(self.log_box.textCursor().End)
        self.log_box.insertPlainText(text)

    def select_config_file(self):
        """Opens a dialog to select the event_config.json file."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Config JSON", "", "JSON Files (*.json)")
        if path:
            self.config_path = path
            self.lbl_config.setText(f"Config: {self.config_path}")

    def select_trace_file(self):
        """Opens a dialog to select the trace.systrace file."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Systrace File", "", "Systrace Files (*.systrace);;All Files (*)")
        if path:
            self.trace_path = path
            self.lbl_trace.setText(f"Trace: {self.trace_path}")

    def add_plot_tab(self, fig, title):
        """Adds a new tab with a matplotlib figure and a save button."""
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        # Save button
        btn_save = QPushButton("Save as PNG")
        def save_png():
            path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", f"{title}.png", "PNG Files (*.png)")
            if path:
                fig.savefig(path)
                print(f"Saved plot to {path}")
        btn_save.clicked.connect(save_png)
        # Layout for tab
        tab_widget = QWidget()
        vbox = QVBoxLayout(tab_widget)
        vbox.addWidget(canvas)
        vbox.addWidget(btn_save)
        self.tabs.addTab(tab_widget, title)

    def add_summary_table_tab(self, summary_stats, events):
        """Adds a new tab with the event summary table."""
        # Define event type order
        event_type_order = ['Task', 'Interrupt', 'Latency', 'ComplexLatency']
        # Map event_name to event_type
        event_type_map = {e['event_name']: e.get('event_type', 'Task') for e in events}
        # Sort summary_stats by event_type order
        def event_sort_key(stat):
            etype = event_type_map.get(stat['event_name'], 'Task')
            return (event_type_order.index(etype) if etype in event_type_order else 99, stat['event_name'])
        sorted_stats = sorted(summary_stats, key=event_sort_key)
        # Count rows and columns
        rows = len(sorted_stats)
        columns = 7  # name, wait avg/min/max, runtime avg/min/max
        # Create table
        table = QTableWidget(rows, columns)
        table.setHorizontalHeaderLabels([
            'Name', 'Wait Avg (ms)', 'Wait Min (ms)', 'Wait Max (ms)',
            'Runtime Avg (ms)', 'Runtime Min (ms)', 'Runtime Max (ms)'])
        # Define row background colors for each event type
        row_bg_colors = {
            'Task': QtGui.QColor('#232629'),            # dark gray
            'Interrupt': QtGui.QColor('#2d3136'),      # slightly lighter dark gray
            'Latency': QtGui.QColor('#1a222d'),        # blue-gray
            'ComplexLatency': QtGui.QColor('#2c2236')  # purple-gray
        }
        for row, stat in enumerate(sorted_stats):
            name = stat['event_name']
            etype = event_type_map.get(name, 'Task')
            row_color = row_bg_colors.get(etype, QtGui.QColor('#232629'))
            # Name
            name_item = QTableWidgetItem(name)
            name_item.setBackground(row_color)
            table.setItem(row, 0, name_item)
            # Wait times (only for Task, ComplexLatency)
            if etype in ['Task', 'ComplexLatency'] and stat['avg_wait'] is not None:
                avg_wait_item = QTableWidgetItem(f"{stat['avg_wait']*1000:.3f}")
                avg_wait_item.setBackground(QtGui.QColor('#444d5c'))
                table.setItem(row, 1, avg_wait_item)
                min_wait_item = QTableWidgetItem(f"{stat['min_wait']*1000:.3f}")
                min_wait_item.setBackground(row_color)
                table.setItem(row, 2, min_wait_item)
                max_wait_item = QTableWidgetItem(f"{stat['max_wait']*1000:.3f}")
                max_wait_item.setBackground(row_color)
                table.setItem(row, 3, max_wait_item)
            else:
                avg_wait_item = QTableWidgetItem("")
                avg_wait_item.setBackground(QtGui.QColor('#444d5c'))
                table.setItem(row, 1, avg_wait_item)
                min_wait_item = QTableWidgetItem("")
                min_wait_item.setBackground(row_color)
                table.setItem(row, 2, min_wait_item)
                max_wait_item = QTableWidgetItem("")
                max_wait_item.setBackground(row_color)
                table.setItem(row, 3, max_wait_item)
            # Runtime (always shown)
            avg_runtime_item = QTableWidgetItem(f"{stat['avg_runtime']*1000:.3f}" if stat['avg_runtime'] is not None else "")
            avg_runtime_item.setBackground(QtGui.QColor('#444d5c'))
            table.setItem(row, 4, avg_runtime_item)
            min_runtime_item = QTableWidgetItem(f"{stat['min_runtime']*1000:.3f}" if stat['min_runtime'] is not None else "")
            min_runtime_item.setBackground(row_color)
            table.setItem(row, 5, min_runtime_item)
            max_runtime_item = QTableWidgetItem(f"{stat['max_runtime']*1000:.3f}" if stat['max_runtime'] is not None else "")
            max_runtime_item.setBackground(row_color)
            table.setItem(row, 6, max_runtime_item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        # Set header style to dark gray
        table.setStyleSheet("QHeaderView::section { background-color: #333333; color: #f0f0f0; font-weight: bold; }")
        
        # Create Excel save button
        btn_save_excel = QPushButton("Save Event Summary to Excel")
        btn_save_excel.setStyleSheet("""
            QPushButton {
                background-color: #2d5aa0;
                color: white;
                border: none;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a6bb8;
            }
            QPushButton:pressed {
                background-color: #1e4a7a;
            }
        """)
        
        def save_excel():
            try:
                from PyQt5.QtWidgets import QFileDialog
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Event Summary Excel", 
                    "event_summary.xlsx", 
                    "Excel Files (*.xlsx)"
                )
                if file_path:
                    # Create DataFrame for Excel
                    excel_data = []
                    for stat in sorted_stats:
                        name = stat['event_name']
                        etype = event_type_map.get(name, 'Task')
                        
                        # Wait times (convert to ms)
                        avg_wait_ms = stat['avg_wait'] * 1000 if stat['avg_wait'] is not None else None
                        min_wait_ms = stat['min_wait'] * 1000 if stat['min_wait'] is not None else None
                        max_wait_ms = stat['max_wait'] * 1000 if stat['max_wait'] is not None else None
                        
                        # Runtime times (convert to ms)
                        avg_runtime_ms = stat['avg_runtime'] * 1000 if stat['avg_runtime'] is not None else None
                        min_runtime_ms = stat['min_runtime'] * 1000 if stat['min_runtime'] is not None else None
                        max_runtime_ms = stat['max_runtime'] * 1000 if stat['max_runtime'] is not None else None
                        
                        excel_data.append({
                            'Event Name': name,
                            'Event Type': etype,
                            'Wait Avg (ms)': f"{avg_wait_ms:.3f}" if avg_wait_ms is not None else "",
                            'Wait Min (ms)': f"{min_wait_ms:.3f}" if min_wait_ms is not None else "",
                            'Wait Max (ms)': f"{max_wait_ms:.3f}" if max_wait_ms is not None else "",
                            'Runtime Avg (ms)': f"{avg_runtime_ms:.3f}" if avg_runtime_ms is not None else "",
                            'Runtime Min (ms)': f"{min_runtime_ms:.3f}" if min_runtime_ms is not None else "",
                            'Runtime Max (ms)': f"{max_runtime_ms:.3f}" if max_runtime_ms is not None else ""
                        })
                    
                    # Create Excel file with multiple sheets
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Main summary sheet
                        summary_df = pd.DataFrame(excel_data)
                        summary_df.to_excel(writer, sheet_name='Event Summary', index=False)
                        
                        # Event type breakdown sheets
                        for etype in event_type_order:
                            etype_data = [row for row in excel_data if row['Event Type'] == etype]
                            if etype_data:
                                etype_df = pd.DataFrame(etype_data)
                                etype_df.to_excel(writer, sheet_name=f'{etype} Events', index=False)
                        
                        # Statistics sheet
                        stats_data = []
                        for etype in event_type_order:
                            etype_stats = [row for row in excel_data if row['Event Type'] == etype]
                            if etype_stats:
                                stats_data.append({
                                    'Event Type': etype,
                                    'Count': len(etype_stats),
                                    'Has Wait Time': any(row['Wait Avg (ms)'] != "" for row in etype_stats),
                                    'Has Runtime': any(row['Runtime Avg (ms)'] != "" for row in etype_stats)
                                })
                        
                        if stats_data:
                            stats_df = pd.DataFrame(stats_data)
                            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                    
                    print(f"Event summary saved to Excel: {file_path}")
                    
            except Exception as e:
                print(f"Error saving Excel file: {str(e)}")
        
        btn_save_excel.clicked.connect(save_excel)
        
        # Add to tab
        tab_widget = QWidget()
        vbox = QVBoxLayout(tab_widget)
        vbox.addWidget(table)
        vbox.addWidget(btn_save_excel)
        self.tabs.addTab(tab_widget, "Event Summary Table")

    def add_latency_table_tab(self, latency_stats):
        """Adds a new tab with the latency summary table."""
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QFileDialog
        from PyQt5 import QtGui
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        table = QTableWidget(tab_widget)
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['Event Name', 'Avg Latency (ms)', 'Min Latency (ms)', 'Max Latency (ms)'])
        table.setRowCount(len(latency_stats))
        row_color = QtGui.QColor('#1a222d')
        text_color = QtGui.QColor('#f0f0f0')
        for row, stat in enumerate(latency_stats):
            name_item = QTableWidgetItem(str(stat['event_name']))
            name_item.setBackground(row_color)
            name_item.setForeground(text_color)
            table.setItem(row, 0, name_item)
            avg_item = QTableWidgetItem(f"{stat['avg_latency']*1000:.3f}" if stat['avg_latency'] is not None else "")
            avg_item.setBackground(row_color)
            avg_item.setForeground(text_color)
            table.setItem(row, 1, avg_item)
            min_item = QTableWidgetItem(f"{stat['min_latency']*1000:.3f}" if stat['min_latency'] is not None else "")
            min_item.setBackground(row_color)
            min_item.setForeground(text_color)
            table.setItem(row, 2, min_item)
            max_item = QTableWidgetItem(f"{stat['max_latency']*1000:.3f}" if stat['max_latency'] is not None else "")
            max_item.setBackground(row_color)
            max_item.setForeground(text_color)
            table.setItem(row, 3, max_item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setStyleSheet("QHeaderView::section { background-color: #333333; color: #f0f0f0; font-weight: bold; }")
        btn_save_excel = QPushButton("Save Latency Summary to Excel")
        def save_excel():
            file_path, _ = QFileDialog.getSaveFileName(tab_widget, "Save Latency Summary Excel", "latency_summary.xlsx", "Excel Files (*.xlsx)")
            if file_path:
                import pandas as pd
                df = pd.DataFrame(latency_stats)
                df[['avg_latency','min_latency','max_latency']] = df[['avg_latency','min_latency','max_latency']] * 1000
                df.to_excel(file_path, index=False)
        btn_save_excel.clicked.connect(save_excel)
        layout.addWidget(table)
        layout.addWidget(btn_save_excel)
        tab_widget.setLayout(layout)
        self.tabs.addTab(tab_widget, "Latency Summary Table")

    def run_analysis(self):
        """Runs the full analysis and plotting workflow."""
        self.log_box.clear()
        self.tabs.clear()
        self.progress_bar.setValue(0)
        QApplication.processEvents() # Update UI

        print("Starting analysis...")
        self.analyzer = SystraceEngine(config_path=self.config_path)
        # Determine max_lines
        try:
            max_lines = int(self.max_lines_edit.text())
        except Exception:
            max_lines = 2000000
        # Count file lines if needed
        try:
            with open(self.trace_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = None
        if total_lines is not None and total_lines < max_lines:
            max_lines = total_lines
        print(f"Reading up to {max_lines} lines from systrace file.")
        self.analyzer.parse_systrace(self.trace_path, max_lines=max_lines)

        all_results = []
        summary_stats = []
        latency_stats = []
        events = self.analyzer.config['events']
        num_events = len(events)
        self.interrupt_results = {}  # event_name별 interrupt 분석 결과 저장
        for idx, event_config in enumerate(events):
            print(f"\nAnalyzing event: {event_config['event_name']}")
            event_type = event_config.get('event_type', 'Task')
            event_df = pd.DataFrame()
            if event_type == 'Task':
                event_df = self.analyzer.analyze_event(event_config)
            elif event_type == 'Latency':
                event_df = self.analyzer.analyze_latency(event_config)
            elif event_type == 'Interrupt':
                event_df = self.analyzer.analyze_interrupt(event_config)
                self.interrupt_results[event_config['event_name']] = event_df
            elif event_type == 'ComplexLatency':
                event_df = self.analyzer.analyze_complex_latency(event_config, self.interrupt_results)
            else:
                print(f"Warning: Unknown event type '{event_type}' for event '{event_config['event_name']}'. Skipping.")
                continue

            all_results.append(event_df)
            if not event_df.empty:
                if event_type == 'Latency':
                    latency_stats.append({
                        'event_name': event_config['event_name'],
                        'avg_latency': event_df['latency'].mean(),
                        'min_latency': event_df['latency'].min(),
                        'max_latency': event_df['latency'].max()
                    })
                else:
                    # waiting_time 통계도 추가
                    if 'waiting_time' in event_df.columns and event_df['waiting_time'].notnull().any():
                        avg_wait = event_df['waiting_time'].mean()
                        min_wait = event_df['waiting_time'].min()
                        max_wait = event_df['waiting_time'].max()
                    else:
                        avg_wait = min_wait = max_wait = None
                    summary_stats.append({
                        'event_name': event_config['event_name'],
                        'avg_runtime': event_df['runtime'].mean() if 'runtime' in event_df.columns else None,
                        'min_runtime': event_df['runtime'].min() if 'runtime' in event_df.columns else None,
                        'max_runtime': event_df['runtime'].max() if 'runtime' in event_df.columns else None,
                        'avg_wait': avg_wait,
                        'min_wait': min_wait,
                        'max_wait': max_wait
                    })
            # Progress bar update
            progress = int((idx + 1) / num_events * 100)
            self.progress_bar.setValue(progress)
            QApplication.processEvents()
        self.all_results = all_results  # 분석 결과를 멤버 변수로 저장
        self.event_names = [e['event_name'] for e in events]
        
        print("\nAnalysis complete. Generating plots...")

        # --- Generate and display plots ---
        # 1. Event Summary Plot (Latency 제외)
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            fig_summary = Figure(figsize=(10, 6))
            plot_event_summary(summary_df, fig_summary)
            self.add_plot_tab(fig_summary, "Event Summary")
            # Add summary table tab
            self.add_summary_table_tab(summary_stats, [e for e in events if e.get('event_type', 'Task') != 'Latency'])

        # 1-2. Latency Summary Plot/Table
        if latency_stats:
            latency_df = pd.DataFrame(latency_stats)
            fig_latency = Figure(figsize=(8, 5))
            plot_latency_summary(latency_df, fig_latency)
            self.add_plot_tab(fig_latency, "Latency Summary")
            self.add_latency_table_tab(latency_stats)

        # 2. Box Plot (event_type별로)
        event_types = ['Task', 'ComplexLatency', 'Interrupt']  # Latency 제외
        for etype in event_types:
            etype_results = [df for df, cfg in zip(all_results, events) if df is not None and not df.empty and cfg.get('event_type', 'Task') == etype]
            if etype_results:
                fig_boxplot = Figure(figsize=(12, 6))
                plot_event_boxplot(etype_results, fig_boxplot)
                self.add_plot_tab(fig_boxplot, f"{etype} Distribution")
        # Latency Distribution Box Plot
        latency_results = [df for df, cfg in zip(all_results, events) if df is not None and not df.empty and cfg.get('event_type', 'Task') == 'Latency']
        if latency_results:
            fig_latency_boxplot = Figure(figsize=(10, 5))
            plot_latency_boxplot(latency_results, fig_latency_boxplot)
            self.add_plot_tab(fig_latency_boxplot, "Latency Distribution")

        # 3. Sequential Plots
        for df in all_results:
            if df is not None and not df.empty:
                event_name = df['event_name'].iloc[0]
                fig_seq = Figure(figsize=(15, 7))
                plot_runtime_over_time(df, fig_seq)
                self.add_plot_tab(fig_seq, f"{event_name} Sequential")
        
        print("\nPlots generated successfully.")
        self.progress_bar.setValue(100)
        QApplication.processEvents()

    def save_results_csv(self):
        """분석 결과를 이벤트별로 각각 CSV 파일로 저장합니다."""
        if not hasattr(self, 'all_results') or not self.all_results:
            print("No analysis results to save.")
            return
        from PyQt5.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory to Save CSV Files")
        if not dir_path:
            return
        for df, name, cfg in zip(self.all_results, getattr(self, 'event_names', []), self.analyzer.config['events']):
            if df is not None and not df.empty:
                event_type = cfg.get('event_type', 'Task')
                # 시간 관련 컬럼 소숫점 6자리로 반올림
                float_cols = [col for col in ['wakeup_time', 'start_time', 'end_time', 'runtime', 'waiting_time', 'mid_time'] if col in df.columns]
                df[float_cols] = df[float_cols].round(6)
                file_path = os.path.join(dir_path, f"{name}.csv")
                if event_type == 'Task':
                    cols = ['event_name','event_type','wakeup_time','start_time','end_time','waiting_time','runtime','cpu','start_process','end_process']
                    save_df = df[[c for c in cols if c in df.columns]]
                    save_df.to_csv(file_path, index=False, float_format='%.6f')
                else:
                    df.to_csv(file_path, index=False, float_format='%.6f')
                print(f"Saved {file_path}")

    def save_results_sqlite(self):
        """분석 결과를 SQLite DB로 저장합니다. 각 이벤트별로 테이블 생성."""
        if not hasattr(self, 'all_results') or not self.all_results:
            print("No analysis results to save.")
            return
        from PyQt5.QtWidgets import QFileDialog
        db_path, _ = QFileDialog.getSaveFileName(self, "Save Results as SQLite DB", "results.db", "SQLite DB Files (*.db)")
        if not db_path:
            return
        import sqlite3
        conn = sqlite3.connect(db_path)
        for df, name in zip(self.all_results, getattr(self, 'event_names', [])):
            if df is not None and not df.empty:
                # 주요 float 컬럼 소수점 9자리로 반올림
                float_cols = [col for col in ['runtime', 'start_time', 'end_time', 'mid_time'] if col in df.columns]
                df[float_cols] = df[float_cols].round(9)
                # 테이블명에 공백/특수문자 방지
                table_name = ''.join(c if c.isalnum() or c=='_' else '_' for c in name)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Saved table {table_name} to {db_path}")
        conn.close()

    def save_results_perfetto_json(self):
        """분석 결과를 Perfetto traceEvents JSON 포맷으로 저장합니다. (event/interrupt만 포함)"""
        if not hasattr(self, 'all_results') or not self.all_results:
            print("No analysis results to save.")
            return
        from PyQt5.QtWidgets import QFileDialog
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Save Results as Perfetto JSON", "perfetto_results.json", "JSON Files (*.json)")
        if not path:
            return
        trace_events = []
        # --- Add metadata event (ph: 'M') ---
        meta_event = {
            "name": "trace_metadata",
            "ph": "M",
            "ts": 0,
            "pid": 0,
            "tid": 0,
            "args": {
                "trace_name": os.path.basename(self.trace_path) if hasattr(self, 'trace_path') else "",
                "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config_version": getattr(self, 'config_path', ''),
                "description": self.analyzer.config.get('description', '') if hasattr(self, 'analyzer') and hasattr(self.analyzer, 'config') else ''
            }
        }
        trace_events.append(meta_event)
        # --- Add process_name, thread_name meta events for each event ---
        pid = 1
        tid = 1
        for df, name, cfg in zip(self.all_results, getattr(self, 'event_names', []), self.analyzer.config['events']):
            event_type = cfg.get('event_type', 'Task')
            if df is not None and not df.empty and event_type in ['Task', 'Interrupt']:
                # process_name
                trace_events.append({
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "args": {"name": name}
                })
                # thread_name
                trace_events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": f"{name} Main"}
                })
                # duration events (Task, Interrupt만)
                float_cols = [col for col in ['runtime', 'start_time', 'end_time', 'mid_time'] if col in df.columns]
                df[float_cols] = df[float_cols].round(9)
                # --- Add FS-FE interval events for Interrupts ---
                if event_type == 'Interrupt' and 'tag' in df.columns:
                    fs_rows = df[df['tag'] == 'FS']
                    fe_rows = df[df['tag'] == 'FE']
                    fs_indices = fs_rows.index.tolist()
                    fe_indices = fe_rows.index.tolist()
                    # Pair FS with next FE (by index order)
                    fe_iter = iter(fe_indices)
                    try:
                        fe_idx = next(fe_iter)
                    except StopIteration:
                        fe_idx = None
                    for fs_idx in fs_indices:
                        # Find the next FE after this FS
                        while fe_idx is not None and fe_idx <= fs_idx:
                            try:
                                fe_idx = next(fe_iter)
                            except StopIteration:
                                fe_idx = None
                        if fe_idx is not None:
                            fs_row = df.loc[fs_idx]
                            fe_row = df.loc[fe_idx]
                            ts_fs = int(fs_row['start_time'] * 1_000_000)
                            ts_fe = int(fe_row['start_time'] * 1_000_000)
                            dur = ts_fe - ts_fs
                            if dur > 0:
                                args_dict = {k: str(fs_row[k]) for k in fs_row.index if k not in ['start_time', 'end_time', 'runtime', 'mid_time']}
                                event = {
                                    "name": f"{name} FS-FE",
                                    "ph": "X",
                                    "ts": ts_fs,
                                    "dur": dur,
                                    "pid": pid,
                                    "tid": tid,
                                    "args": args_dict
                                }
                                trace_events.append(event)
                # --- Add original events as before ---
                for _, row in df.iterrows():
                    ts = int(row['start_time'] * 1_000_000)  # s -> us
                    dur = int(row['runtime'] * 1_000_000)    # s -> us
                    if dur <= 0:
                        continue  # duration이 0 이하인 이벤트는 무시
                    args_dict = {k: str(row[k]) for k in row.index if k not in ['start_time', 'end_time', 'runtime', 'mid_time']}
                    if 'frame_number' in row.index and pd.notnull(row['frame_number']):
                        args_dict['frame_number'] = int(row['frame_number'])
                    event = {
                        "name": name,
                        "ph": "X",
                        "ts": ts,
                        "dur": dur,
                        "pid": pid,
                        "tid": tid,
                        "args": args_dict
                    }
                    trace_events.append(event)
                pid += 1  # 이벤트별로 pid 증가
        out = {"traceEvents": trace_events}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved Perfetto JSON to {path}")


# --- Plotting Functions (Refactored for UI) ---

def plot_event_summary(summary_df: pd.DataFrame, fig: Figure):
    """Plots a bar chart of min/avg/max waiting times and runtimes on a given Figure (wait first)."""
    fig.clear()
    ax = fig.add_subplot(111)
    summary_df = summary_df.copy()
    summary_df['min_runtime'] = summary_df['min_runtime'] * 1000
    summary_df['avg_runtime'] = summary_df['avg_runtime'] * 1000
    summary_df['max_runtime'] = summary_df['max_runtime'] * 1000
    if 'min_wait' in summary_df.columns:
        summary_df['min_wait'] = summary_df['min_wait'] * 1000
        summary_df['avg_wait'] = summary_df['avg_wait'] * 1000
        summary_df['max_wait'] = summary_df['max_wait'] * 1000
    x = np.arange(len(summary_df['event_name']))
    width = 0.13
    # waiting time bars (wait first)
    if 'min_wait' in summary_df.columns:
        minw_bars = ax.bar(x - width*2, summary_df['min_wait'], width, label='Min Wait', edgecolor='darkred', color='mistyrose')
        avgw_bars = ax.bar(x - width, summary_df['avg_wait'], width, label='Avg Wait', edgecolor='darkred', color='salmon')
        maxw_bars = ax.bar(x, summary_df['max_wait'], width, label='Max Wait', edgecolor='darkred', color='red')
    # runtime bars (runtime after wait)
    min_bars = ax.bar(x + width, summary_df['min_runtime'], width, label='Min Runtime', edgecolor='navy', color='skyblue')
    avg_bars = ax.bar(x + width*2, summary_df['avg_runtime'], width, label='Avg Runtime', edgecolor='navy', color='dodgerblue')
    max_bars = ax.bar(x + width*3, summary_df['max_runtime'], width, label='Max Runtime', edgecolor='navy', color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['event_name'])
    ax.set_ylabel('Time (msec)')
    ax.set_title('Event Waiting Time & Runtime Summary')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    if 'min_wait' in summary_df.columns:
        for bars in [minw_bars, avgw_bars, maxw_bars]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=8, color='darkred')
    for bars in [min_bars, avg_bars, max_bars]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()


def plot_event_boxplot(all_results: List[pd.DataFrame], fig: Figure):
    """Plots a boxplot of waiting time and runtime distributions on a given Figure (wait first)."""
    fig.clear()
    ax = fig.add_subplot(111)
    data, labels, stats, colors, box_colors, event_colors = [], [], [], [], [], {}
    # 고유 색상 리스트
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_idx = 0
    for df in all_results:
        if df is not None and len(df) > 0:
            event_name = df['event_name'].iloc[0]
            # event별 고유 색상 지정
            if event_name not in event_colors:
                event_colors[event_name] = color_list[color_idx % len(color_list)]
                color_idx += 1
            event_color = event_colors[event_name]
            # waiting_time (wait first)
            if 'waiting_time' in df.columns and df['waiting_time'].notnull().any():
                waits_ms = df['waiting_time'].dropna().values * 1000
                if len(waits_ms) > 0:
                    data.append(waits_ms)
                    labels.append(event_name + ' (wait)')
                    stats.append({'avg': waits_ms.mean(), 'min': waits_ms.min(), 'max': waits_ms.max()})
                    box_colors.append('darkred')
                    colors.append(event_color)
            # runtime (runtime after wait)
            runtimes_ms = df['runtime'].values * 1000
            data.append(runtimes_ms)
            labels.append(event_name + ' (runtime)')
            stats.append({'avg': runtimes_ms.mean(), 'min': runtimes_ms.min(), 'max': runtimes_ms.max()})
            box_colors.append('navy')
            colors.append(event_color)
    if not data: return
    box = ax.boxplot(data, patch_artist=True, labels=labels, showmeans=True, meanline=True)
    # box color
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
        patch.set_facecolor('white')
    # whisker color
    for whisker, color in zip(box['whiskers'], [c for c in box_colors for _ in (0,1)]):
        whisker.set_color(color)
        whisker.set_linewidth(2)
    # cap color
    for cap, color in zip(box['caps'], [c for c in box_colors for _ in (0,1)]):
        cap.set_color(color)
        cap.set_linewidth(2)
    # median color
    for median, color in zip(box['medians'], box_colors):
        median.set_color(color)
        median.set_linewidth(2)
    # mean color
    for mean, color in zip(box['means'], box_colors):
        mean.set_markerfacecolor(color)
        mean.set_markeredgecolor(color)
    ax.set_ylabel('Time (msec)')  # Y축 단위 명확히 표기
    ax.set_title('Event Waiting Time & Runtime Distribution')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    # annotate
    for i, stat in enumerate(stats, 1):
        ax.text(i, stat['max'], f"max={stat['max']:.3f}", ha='center', va='bottom', fontsize=8)
        ax.text(i, stat['min'], f"min={stat['min']:.3f}", ha='center', va='top', fontsize=8)
        ax.text(i, stat['avg'], f"avg={stat['avg']:.3f}", ha='center', va='bottom', fontsize=8, color='red')
    # legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(edgecolor='darkred', facecolor='white', linewidth=2, label='Waiting Time'),
        Patch(edgecolor='navy', facecolor='white', linewidth=2, label='Runtime')
    ]
    ax.legend(handles=legend_handles)
    # x축 라벨 색상 지정
    ax.set_xticklabels(labels, rotation=15)
    for ticklabel, color in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(color)
    fig.tight_layout()


def plot_runtime_over_time(df: pd.DataFrame, fig: Figure):
    """Plots the runtime (or latency) of a single event type over time on a given Figure."""
    fig.clear()
    ax = fig.add_subplot(111)

    event_name = df['event_name'].iloc[0]
    # runtime 또는 latency 컬럼 자동 선택
    if 'runtime' in df.columns:
        yvals_ms = df['runtime'] * 1000
        ylabel = "Runtime (msec)"
        title = f"Runtime Over Time for '{event_name}'"
    elif 'latency' in df.columns:
        yvals_ms = df['latency'] * 1000
        ylabel = "Latency (msec)"
        title = f"Latency Over Time for '{event_name}'"
    else:
        return  # 표시할 값이 없음

    start_times_ms = df['start_time'] * 1000 if 'start_time' in df.columns else range(len(df))

    ax.plot(start_times_ms, yvals_ms, marker='o', linestyle='-')

    for start_ms, yval in zip(start_times_ms, yvals_ms):
        ax.text(start_ms, yval, f' {yval:.3f}', fontsize=8, ha='left', va='bottom')

    ax.set_xlabel("Start Time (msec)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    fig.tight_layout()


# --- Latency summary plot/table helpers ---
def plot_latency_summary(latency_df: pd.DataFrame, fig: Figure):
    """Plots a bar chart of min/avg/max latency for Latency events only."""
    fig.clear()
    ax = fig.add_subplot(111)
    latency_df = latency_df.copy()
    latency_df['min_latency'] = latency_df['min_latency'] * 1000
    latency_df['avg_latency'] = latency_df['avg_latency'] * 1000
    latency_df['max_latency'] = latency_df['max_latency'] * 1000
    x = np.arange(len(latency_df['event_name']))
    width = 0.2
    min_bars = ax.bar(x - width, latency_df['min_latency'], width, label='Min Latency', color='skyblue')
    avg_bars = ax.bar(x, latency_df['avg_latency'], width, label='Avg Latency', color='dodgerblue')
    max_bars = ax.bar(x + width, latency_df['max_latency'], width, label='Max Latency', color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels(latency_df['event_name'])
    ax.set_ylabel('Latency (msec)')
    ax.set_title('Latency Event Summary')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    for bars in [min_bars, avg_bars, max_bars]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()


def plot_latency_boxplot(latency_results: List[pd.DataFrame], fig: Figure):
    """Plots a boxplot of latency distributions for Latency events only."""
    fig.clear()
    ax = fig.add_subplot(111)
    data, labels, stats, colors, box_colors, event_colors = [], [], [], [], [], {}
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_idx = 0
    for df in latency_results:
        if df is not None and len(df) > 0 and 'latency' in df.columns:
            event_name = df['event_name'].iloc[0]
            if event_name not in event_colors:
                event_colors[event_name] = color_list[color_idx % len(color_list)]
                color_idx += 1
            event_color = event_colors[event_name]
            latencies_ms = df['latency'].values * 1000
            data.append(latencies_ms)
            labels.append(event_name + ' (latency)')
            stats.append({'avg': latencies_ms.mean(), 'min': latencies_ms.min(), 'max': latencies_ms.max()})
            box_colors.append('navy')
            colors.append(event_color)
    if not data: return
    box = ax.boxplot(data, patch_artist=True, labels=labels, showmeans=True, meanline=True)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
        patch.set_facecolor('white')
    for whisker, color in zip(box['whiskers'], [c for c in box_colors for _ in (0,1)]):
        whisker.set_color(color)
        whisker.set_linewidth(2)
    for cap, color in zip(box['caps'], [c for c in box_colors for _ in (0,1)]):
        cap.set_color(color)
        cap.set_linewidth(2)
    for median, color in zip(box['medians'], box_colors):
        median.set_color(color)
        median.set_linewidth(2)
    for mean, color in zip(box['means'], box_colors):
        mean.set_markerfacecolor(color)
        mean.set_markeredgecolor(color)
    ax.set_ylabel('Latency (msec)')
    ax.set_title('Latency Event Distribution')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    for i, stat in enumerate(stats, 1):
        ax.text(i, stat['max'], f"max={stat['max']:.3f}", ha='center', va='bottom', fontsize=8)
        ax.text(i, stat['min'], f"min={stat['min']:.3f}", ha='center', va='top', fontsize=8)
        ax.text(i, stat['avg'], f"avg={stat['avg']:.3f}", ha='center', va='bottom', fontsize=8, color='red')
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(edgecolor='navy', facecolor='white', linewidth=2, label='Latency')
    ]
    ax.legend(handles=legend_handles)
    ax.set_xticklabels(labels, rotation=15)
    for ticklabel, color in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(color)
    fig.tight_layout()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PerfettoAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())
