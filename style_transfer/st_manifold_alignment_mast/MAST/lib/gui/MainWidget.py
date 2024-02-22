import os
import traceback

from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLabel, QSpinBox, QFileDialog, \
    QRadioButton, QButtonGroup
from torchvision.utils import save_image
from lib.gui.PaintBoard import PaintBoard
from lib.gui.MastServer import MastServer


class MainWidget(QWidget):
    def __init__(self, config, Parent=None):
        super().__init__(Parent)
        self.config = config
        self.mast_server = MastServer(config)
        self.__InitData()  # Initialize the data first, then initialize the interface
        self.__InitView()

    def __InitData(self):
        """
        Initialize member variables
        """
        self.__content_board = PaintBoard(self.config, self)
        self.__style_board = PaintBoard(self.config, self)
        self.__result_board = PaintBoard(self.config, self)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        self.setWindowTitle('MAST-GUI')
        self.setStatusTip('123')

        # Set the horizontal layout as the main window layout
        main_layout = QHBoxLayout(self)

        # Set up a vertical sub-layout and put the function keys on the left
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)

        self.__upContentBtn = QPushButton('Upload content image')
        self.__upContentBtn.clicked.connect(self.__content_board.set_img)
        left_layout.addWidget(self.__upContentBtn)

        self.__upStyleBtn = QPushButton('Upload style image')
        self.__upStyleBtn.clicked.connect(self.__style_board.set_img)
        left_layout.addWidget(self.__upStyleBtn)

        mask_layout = QHBoxLayout()
        self.__label_mask_type = QLabel(self)
        self.__label_mask_type.setText("Mask type")
        mask_layout.addWidget(self.__label_mask_type)
        self.rb1 = QRadioButton('pre')
        self.rb2 = QRadioButton('post')
        if self.config.TEST.GUI.ADD_MASK_TYPE == 'pre':
            self.rb1.setChecked(True)
        else:
            self.rb2.setChecked(True)
        mask_layout.addWidget(self.rb1)
        mask_layout.addWidget(self.rb2)
        self.bg1 = QButtonGroup(self)
        self.bg1.addButton(self.rb1, id=1)
        self.bg1.addButton(self.rb2, id=2)
        self.bg1.buttonClicked.connect(self.rbclicked)
        left_layout.addLayout(mask_layout)

        expand_layout = QHBoxLayout()
        self.__label_expand_type = QLabel(self)
        self.__label_expand_type.setText("Expand")
        expand_layout.addWidget(self.__label_expand_type)
        self.rb3 = QRadioButton('yes')
        self.rb4 = QRadioButton('no')
        if self.config.TEST.GUI.EXPAND:
            self.rb3.setChecked(True)
        else:
            self.rb4.setChecked(True)
        expand_layout.addWidget(self.rb3)
        expand_layout.addWidget(self.rb4)
        self.bg2 = QButtonGroup(self)
        self.bg2.addButton(self.rb3, id=3)
        self.bg2.addButton(self.rb4, id=4)
        self.bg2.buttonClicked.connect(self.rbclicked)
        left_layout.addLayout(expand_layout)

        expand_num_layout = QHBoxLayout()
        self.__label_expand_num = QLabel(self)
        self.__label_expand_num.setText("Expand num")
        self.__label_expand_num.setFixedHeight(20)
        expand_num_layout.addWidget(self.__label_expand_num)
        # left_layout.addWidget(self.__label_penThickness)

        self.__spinBox_expand_num = QSpinBox(self)
        self.__spinBox_expand_num.setMaximum(50)
        self.__spinBox_expand_num.setMinimum(0)
        self.__spinBox_expand_num.setValue(self.config.TEST.GUI.EXPAND_NUM)
        self.__spinBox_expand_num.setSingleStep(2)
        self.__spinBox_expand_num.valueChanged.connect(self.change_expand_num)
        expand_num_layout.addWidget(self.__spinBox_expand_num)
        left_layout.addLayout(expand_num_layout)

        reset_layout = QHBoxLayout()
        self.__resetContentBtn = QPushButton('Reset content')
        self.__resetContentBtn.clicked.connect(self.__content_board.reset)
        reset_layout.addWidget(self.__resetContentBtn)
        self.__resetStyleBtn = QPushButton('Reset style')
        self.__resetStyleBtn.clicked.connect(self.__style_board.reset)
        reset_layout.addWidget(self.__resetStyleBtn)
        self.__resetResultBtn = QPushButton('Reset result')
        self.__resetResultBtn.clicked.connect(self.__result_board.reset)
        reset_layout.addWidget(self.__resetResultBtn)
        left_layout.addLayout(reset_layout)

        self.__resetBtn = QPushButton('Reset all')
        self.__resetBtn.clicked.connect(self.__content_board.reset)
        self.__resetBtn.clicked.connect(self.__style_board.reset)
        self.__resetBtn.clicked.connect(self.__result_board.reset)
        left_layout.addWidget(self.__resetBtn)

        width_color_layout = QHBoxLayout()
        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("Brush width")
        self.__label_penThickness.setFixedHeight(20)
        width_color_layout.addWidget(self.__label_penThickness)
        # left_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(50)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(20)  # The default thickness is 10
        self.__spinBox_penThickness.setSingleStep(2)  # The minimum change value is 2
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)
        width_color_layout.addWidget(self.__spinBox_penThickness)
        # left_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("Color")
        self.__label_penColor.setFixedHeight(20)
        width_color_layout.addWidget(self.__label_penColor)
        # left_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)  # Fill the drop-down list with various colors
        self.__comboBox_penColor.currentIndexChanged.connect(self.on_PenColorChange)
        width_color_layout.addWidget(self.__comboBox_penColor)
        # left_layout.addWidget(self.__comboBox_penColor)
        left_layout.addLayout(width_color_layout)

        self.__startBtn = QPushButton('Start')
        self.__startBtn.clicked.connect(self.start_transfer)
        left_layout.addWidget(self.__startBtn)

        self.__saveBtn = QPushButton('Save result')
        self.__saveBtn.clicked.connect(self.saveImg)
        left_layout.addWidget(self.__saveBtn)

        self.__saveBtn_all = QPushButton('Save all')
        self.__saveBtn_all.clicked.connect(self.saveImg_all)
        left_layout.addWidget(self.__saveBtn_all)

        # Add the left vertical layout to the main layout
        main_layout.addLayout(left_layout)

        main_layout.addWidget(self.__content_board)
        main_layout.addWidget(self.__style_board)
        main_layout.addWidget(self.__result_board)

    def __fillColorList(self, comboBox):
        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__content_board.ChangePenColor(color_str)
        self.__style_board.ChangePenColor(color_str)
        self.__result_board.ChangePenColor(color_str)
        print(f'change brush color to {color_str}')

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__content_board.ChangePenThickness(penThickness)
        self.__style_board.ChangePenThickness(penThickness)
        self.__result_board.ChangePenThickness(penThickness)
        print(f'change brush width to {penThickness}')

    def change_expand_num(self):
        value = self.__spinBox_expand_num.value()
        self.config.TEST.GUI.EXPAND_NUM = value
        print(f'change expand num to {self.config.TEST.GUI.EXPAND_NUM}')

    def rbclicked(self):
        sender = self.sender()
        if sender == self.bg1:
            if self.bg1.checkedId() == 1:
                self.config.TEST.GUI.ADD_MASK_TYPE = 'pre'
                print(f'change add mask type to {self.config.TEST.GUI.ADD_MASK_TYPE}')
            elif self.bg1.checkedId() == 2:
                self.config.TEST.GUI.ADD_MASK_TYPE = 'post'
                print(f'change add mask type to {self.config.TEST.GUI.ADD_MASK_TYPE}')
        elif sender == self.bg2:
            if self.bg2.checkedId() == 3:
                self.config.TEST.GUI.EXPAND = True
                print(f'change expand to {self.config.TEST.GUI.EXPAND}')
            elif self.bg2.checkedId() == 4:
                self.config.TEST.GUI.EXPAND = False
                print(f'change expand to {self.config.TEST.GUI.EXPAND}')

    def saveImg(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath[0])
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__result_board.GetContentAsQImage()
        image.save(savePath[0])

    def saveImg_all(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        if savePath[0] == "":
            print("Save cancel")
            return
        c_img = self.__content_board.GetContentAsQImage()
        s_img = self.__style_board.GetContentAsQImage()
        r_img = self.__result_board.GetContentAsQImage()
        base_path = os.path.splitext(savePath[0])[0]
        c_img.save(f'{base_path}_content.png')
        s_img.save(f'{base_path}_style.png')
        r_img.save(f'{base_path}_res.png')
        print(f'Results have saved in [{base_path}], include content, style and stylized results...')

    def start_transfer(self):
        content_path = self.__content_board.getImgPath()
        style_path = self.__style_board.getImgPath()
        if not content_path:
            print('Please upload the content image!')
            return
        if not style_path:
            print('Please upload the style image!')
            return
        c_pos_dict = self.__content_board.getPosDict()
        s_pos_dict = self.__style_board.getPosDict()
        try:
            out_tensor = self.mast_server.process(content_path, style_path, c_pos_dict, s_pos_dict,
                                                  self.config.TEST.GUI.ADD_MASK_TYPE, self.config.TEST.GUI.EXPAND,
                                                  self.config.TEST.GUI.EXPAND_NUM)
        except Exception as e:
            traceback.print_exc()
            print(e)
            print(f'Sorry, please try again!')
            return
        res_path = os.path.join(self.config.TEST.GUI.TEMP_DIR, 'res.png')
        save_image(out_tensor, res_path)
        self.__result_board.set_already_img(res_path)

    def calWMat(self):
        cPosDict = self.__content_board.getPosDict()
        sPosDict = self.__style_board.getPosDict()
        for cKey in cPosDict.keys():
            cPosSet = cPosDict[cKey]
            sPosSet = sPosDict[cKey]
            for cIndex in cPosSet:
                for sIndex in sPosSet:
                    print(f'({cIndex}, {sIndex})')
                    self.__wMat[cIndex][sIndex] = 1
                    self.__c_pos_list.append(cIndex)
