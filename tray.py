# Standard library imports
import platform
import sys

# Third-party imports (conditionally imported on Windows)
if platform.system() == "Windows":
    try:
        from PySide6.QtCore import QObject, Qt, Signal
        from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
        from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
        PYSIDE6_AVAILABLE = True
    except ImportError:
        PYSIDE6_AVAILABLE = False
else:
    # Create stub classes for non-Windows platforms
    PYSIDE6_AVAILABLE = False
    
    class QObject:
        def __init__(self, parent=None):
            pass
    
    class Signal:
        def __init__(self):
            pass
        
        def emit(self):
            pass


class SystemTray(QObject):
    """System tray handler class"""
    
    # Signals to communicate with main window
    show_window = Signal()
    hide_window = Signal()
    quit_application = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.tray_icon = None
        self.is_windows = platform.system() == "Windows"
        
    def setup(self, log_callback=None):
        """Setup system tray functionality"""
        self.log_callback = log_callback
        
        # On non-Windows platforms, just log and return False
        if not self.is_windows or not PYSIDE6_AVAILABLE:
            if self.log_callback:
                if not self.is_windows:
                    self.log_callback("System tray functionality not available on this platform")
                else:
                    self.log_callback("PySide6 not available, system tray disabled")
            return False
        
        if not QSystemTrayIcon.isSystemTrayAvailable():
            if self.log_callback:
                self.log_callback("System tray is not available on this system")
            return False
            
        # Create a simple icon for the system tray
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(70, 130, 180))  # Steel blue color
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "R")
        painter.end()
        
        icon = QIcon(pixmap)
        
        self.tray_icon = QSystemTrayIcon(icon, self.parent_window)
        self.tray_icon.setToolTip("Route Squiggler - Render Client")
        
        # Create tray menu
        tray_menu = QMenu()
        
        show_action = QAction("Show", self.parent_window)
        show_action.triggered.connect(self.show_window.emit)
        tray_menu.addAction(show_action)
        
        hide_action = QAction("Hide", self.parent_window)
        hide_action.triggered.connect(self.hide_window.emit)
        tray_menu.addAction(hide_action)
        
        tray_menu.addSeparator()
        
        quit_action = QAction("Quit", self.parent_window)
        quit_action.triggered.connect(self.quit_application.emit)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._tray_icon_activated)
        self.tray_icon.show()
        
        if self.log_callback:
            self.log_callback("System tray icon created")
        
        return True
    
    def _tray_icon_activated(self, reason):
        """Handle tray icon activation"""
        if not self.is_windows or not PYSIDE6_AVAILABLE:
            return
            
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.parent_window and self.parent_window.isVisible():
                self.hide_window.emit()
            else:
                self.show_window.emit()
    
    def hide_tray_icon(self):
        """Hide the tray icon"""
        if self.tray_icon and self.is_windows and PYSIDE6_AVAILABLE:
            self.tray_icon.hide()
    
    def is_available(self):
        """Check if system tray is available"""
        if not self.is_windows or not PYSIDE6_AVAILABLE:
            return False
        return QSystemTrayIcon.isSystemTrayAvailable()
    
    def is_visible(self):
        """Check if tray icon is visible"""
        if not self.is_windows or not PYSIDE6_AVAILABLE:
            return False
        return self.tray_icon and self.tray_icon.isVisible() 