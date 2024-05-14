import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QAction, QPainter, QPen, QImage, QPixmap, QColor
from PyQt6.QtCore import QSize, Qt, QPoint
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageQt

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage(280, 280, QImage.Format.Format_Grayscale8)
        self.image.fill(Qt.GlobalColor.white)
        self.last_point = QPoint()
        self.pen_color = QColor(Qt.GlobalColor.black)
        self.pen_width = 20

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def resizeEvent(self, event):
        if self.width() != self.image.width() or self.height() != self.image.height():
            new_width = self.width()
            new_height = self.height()
            self.resizeImage(self.image, QSize(new_width, new_height))

    def resizeImage(self, image, new_size):
        if image.size() == new_size:
            return
        new_image = QImage(new_size, QImage.Format.Format_RGB32)
        new_image.fill(Qt.GlobalColor.white)
        painter = QPainter(new_image)
        painter.drawImage(QPoint(0, 0), image)
        self.image = new_image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digit Recognizer")
        self.setGeometry(100, 100, 280, 280)
        self.drawing_widget = DrawingWidget()
        self.setCentralWidget(self.drawing_widget)
        self.model = GRUNet(input_size=28, hidden_size=128, num_layers=2, num_classes=10)
        self.model.load_state_dict(torch.load("GRU.pth"))
        self.model.eval()
        self.result_label = QLabel("Draw a digit")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.statusBar().addWidget(self.result_label)
        recognize_button = QPushButton("Recognize", self)
        recognize_button.clicked.connect(self.recognizeDigit)
        clear_action = QPushButton("Clear", self)
        clear_action.clicked.connect(self.clearDrawing)
        toolbar = self.addToolBar("Tools")
        toolbar.addWidget(clear_action)
        toolbar.addWidget(recognize_button)

    def recognizeDigit(self):
        image = self.drawing_widget.image
        tensor = convertToTensor(image)
        with torch.no_grad():
            prediction = self.model(tensor)
        print(prediction[0])
        digit = torch.argmax(prediction).item()
        print(digit)
        self.result_label.setText(f"Predicted digit: {digit}")
        self.drawing_widget.update()

    def clearDrawing(self):
        self.drawing_widget.image.fill(Qt.GlobalColor.white)
        self.result_label.setText("Draw a digit")
        self.drawing_widget.update()

def convertToTensor(image):
    scaled_image = image.scaled(28, 28)
    pil_image = ImageQt.fromqimage(scaled_image)
    
    transform = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform(pil_image)
    normalized_tensor = img_tensor.type('torch.IntTensor') / 255 # Нормализация
    # Замена чёрного цвета 0 -> 1, белого 1 -> 0
    normalized_tensor = normalized_tensor - 1 
    normalized_tensor = normalized_tensor.type('torch.FloatTensor') * (-1)
    return normalized_tensor

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())