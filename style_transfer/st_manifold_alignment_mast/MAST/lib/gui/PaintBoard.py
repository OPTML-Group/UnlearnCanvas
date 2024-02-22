from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPen, QColor, QSize
from PyQt5.QtCore import Qt


class PaintBoard(QWidget):
    def __init__(self, config, Parent=None):
        super().__init__(Parent)
        self.config = config
        self.__InitData()
        self.__InitView()

    def __InitData(self):

        self.__size = QSize(self.config.TEST.GUI.IMAGE_SIZE, self.config.TEST.GUI.IMAGE_SIZE)

        self.__imgPath = None

        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.white)

        self.__IsEmpty = True
        self.EraserMode = False

        self.__lastPos = QPoint(0, 0)
        self.__currentPos = QPoint(0, 0)

        self.__painter = QPainter()

        self.__thickness = 10
        self.__penColor = QColor("black")
        self.__colorStr = 'black'

        self.__posDict = {}

    def __InitView(self):
        self.setFixedSize(self.__size)

    def set_img(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "Open Image", "", "All Files(*);;*.jpg;;*.png")
        if imgName:
            jpg = QPixmap(imgName).scaled(self.__size)
            self.__board = jpg
            self.__imgPath = imgName

    def set_already_img(self, path):
        jpg = QPixmap(path).scaled(self.__size)
        self.__board = jpg
        self.update()
        self.__imgPath = path

    def reset(self):
        self.__board.fill(Qt.white)
        self.__imgPath = None
        self.__posDict.clear()
        self.update()
        print(f'Clear img path and pos dict!')

    def getImgPath(self):
        return self.__imgPath

    def getPosDict(self):
        return self.__posDict

    def Clear(self):
        self.__board.fill(Qt.white)
        self.update()
        self.__IsEmpty = True

    def ChangePenColor(self, color="black"):
        self.__penColor = QColor(color)
        self.__colorStr = color
        self.__cap = Qt.RoundCap

    def ChangePenThickness(self, thickness=10):
        self.__thickness = thickness

    def IsEmpty(self):
        return self.__IsEmpty

    def GetContentAsQImage(self):
        image = self.__board.toImage()
        return image

    def paintEvent(self, paintEvent):
        self.__painter.begin(self)
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.add_pos(self.__currentPos)
        self.__painter.begin(self.__board)

        if self.EraserMode == False:
            pen = QPen(self.__penColor, self.__thickness)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            self.__painter.setPen(pen)
        else:
            self.__painter.setPen(QPen(Qt.white, 10))

        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False

    def resetPos(self, currentPos):
        maxpos = int(self.config.TEST.GUI.IMAGE_SIZE - 1)
        if currentPos.x() > maxpos:
            currentPos = QPoint(maxpos, currentPos.y())
        if currentPos.x() < 0:
            currentPos = QPoint(0, currentPos.y())
        if currentPos.y() > maxpos:
            currentPos = QPoint(currentPos.x(), maxpos)
        if currentPos.y() < 0:
            currentPos = QPoint(currentPos.x(), 0)
        res = (currentPos.y(), currentPos.x())
        return res

    def calPosset(self, currentPos):
        maxpos = int(self.config.TEST.GUI.IMAGE_SIZE - 1)
        step = int(self.__thickness / 2)
        newPos = self.resetPos(currentPos)
        res = set()
        for i in range(newPos[0] - step, newPos[0] + step):
            if i > maxpos:
                break
            if i < 0:
                continue
            for j in range(newPos[1] - step, newPos[1] + step):
                if j > maxpos:
                    break
                if j < 0:
                    continue
                res.add((i, j))
        return res

    def add_pos(self, currentPos):
        posSet = self.calPosset(currentPos)
        if self.__colorStr in self.__posDict:
            self.__posDict[self.__colorStr] = self.__posDict[self.__colorStr] | posSet
        else:
            self.__posDict[self.__colorStr] = posSet
