from PyQt5.QtWidgets import QWidget


class PopWidget(QWidget):
    def __init__(self):
        super(PopWidget, self).__init__()
        self.setupUi(self)

    def setupUi(self, PopWidget):
        PopWidget.resize(400, 300)
        PopWidget.setWindowTitle('弹出窗口')



    def show(self):
        super(PopWidget, self).show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        self.hide()
        event.ignore()

