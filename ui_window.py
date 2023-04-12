import sys
from ctypes.wintypes import HWND

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QMdiSubWindow




from view import view
import uinew
import demo
import ui

global class_name
global index




def showview():

    print(ui.comboBox.currentText())
    print(ui.comboBox_2.currentText())
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(),ui.comboBox_2.currentText(),flag,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())
def max():
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(), ui.comboBox_2.currentText(),flag,size=109,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())

def min():
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(), ui.comboBox_2.currentText(),flag,size=110,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())
def original():
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(), ui.comboBox_2.currentText(),flag,size=114,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())

def save():
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(), ui.comboBox_2.currentText(),flag,size=115,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())

def exit():
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(), ui.comboBox_2.currentText(),flag,size=99,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())
    sys.exit(app.exec_())

def color():
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(), ui.comboBox_2.currentText(),flag,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())

def p_size():
    if ui.radioButton.isChecked():
        flag=0
    else:
        flag=1
    view(ui.comboBox.currentText(), ui.comboBox_2.currentText(),flag,color=ui.horizontalSlider_2.value(),p_size=ui.horizontalSlider.value())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = ui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    #MainWindow.showFullScreen()

    MainWindow.show()
    ui.pushButton_3.clicked.connect(showview)
    ui.pushButton.clicked.connect(max)
    ui.pushButton_2.clicked.connect(min)
    ui.pushButton_6.clicked.connect(original)
    ui.pushButton_4.clicked.connect(save)
    ui.pushButton_5.clicked.connect(exit)
    ui.horizontalSlider_2.valueChanged.connect(color)
    ui.horizontalSlider.valueChanged.connect(p_size)
    sys.exit(app.exec_())
