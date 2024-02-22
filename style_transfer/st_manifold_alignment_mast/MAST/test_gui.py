import os
import sys
import argparse
from PyQt5.QtWidgets import QApplication
from lib.gui.MainWidget import MainWidget
from lib.core.config import get_cfg


def main():
    parser = argparse.ArgumentParser(description='GUI Test')
    parser.add_argument('--cfg_path', type=str, default='configs/config.yaml',
                        help='config path')
    args = parser.parse_args()
    cfg = get_cfg(cfg_path=args.cfg_path)
    os.makedirs(cfg.TEST.GUI.TEMP_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    mainWidget = MainWidget(cfg)  # build a new main widget
    mainWidget.show()  # show the main widget
    exit(app.exec_())  # start message loop


if __name__ == '__main__':
    main()
