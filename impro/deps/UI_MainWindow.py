import os

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QAction, QLabel
from PyQt5.QtGui import QIcon


# 获取当前文件的路径
THIS_PATH = os.path.dirname(os.path.abspath(__file__))

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.centralwidget = None
        self.menu_file = None
        self.setupUi(self)

    def setupUi(self, MainWindow):
        self.centralwidget = MainWindow
        # 设置窗口标题
        self.centralwidget.setWindowTitle('Image Process Programing Platform')
        
        # 设置窗口位置和大小
        w = 1680
        h = 960
        self.centralwidget.setGeometry(0, 0, w, h)  # 设置窗口左上角坐标为(0, 0)，宽、高分别为w、h

        # 设置应用程序图标
        icon = QIcon()
        icon.addFile(os.path.join(os.path.dirname(THIS_PATH), 'assets/appicon.svg'), QSize(30, 30), QIcon.Normal, QIcon.Off)
        self.centralwidget.setWindowIcon(icon)

        # 菜单栏
        self.menu_bar = QMenuBar()
        self.menu_bar.setParent(self.centralwidget)
        self.menu_bar.setGeometry(0, 20, 1440, 30)  # 若后续增加的菜单太多，会显示不完全，更改第三个参数即可

        # -*-*-*-*-*-**-*-*     文件菜单     *-*-*-*-*-*-*-*-* #
        self.menu_file = QMenu('文件', self.centralwidget)
        self.act_open = QAction('打开', self.centralwidget)
        self.act_open_folder = QAction('打开目录', self.centralwidget)
        self.act_close = QAction('关闭', self.centralwidget)
        self.act_save = QAction('保存',self.centralwidget)
        self.act_quit = QAction('退出', self.centralwidget)
        self.act_prev = QAction('上一张', self.centralwidget)
        self.act_next = QAction('下一张', self.centralwidget)
        self.menu_file.addAction(self.act_open)
        self.menu_file.addAction(self.act_open_folder)
        self.menu_file.addAction(self.act_close)
        self.menu_file.addAction(self.act_save)
        self.menu_file.addAction(self.act_quit)
        self.menu_file.addAction(self.act_prev)
        self.menu_file.addAction(self.act_next)
        # -*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-* #

        self.act_plot_hist = QAction('绘制直方图', self.centralwidget)

        # -*-*-*-*-*-**-*     灰度变换菜单     *-*-*-*-*-*-*-* #
        self.menu_gray_trans = QMenu('灰度变换', self.centralwidget)
        self.act_binarize = QAction('二值化', self.centralwidget)
        self.act_exp_trans = QAction('幂次变换', self.centralwidget)
        self.act_log_trans = QAction('对数变换', self.centralwidget)
        self.act_hist_eql = QAction('直方图均衡', self.centralwidget)
        self.menu_gray_trans.addAction(self.act_binarize)
        self.menu_gray_trans.addAction(self.act_exp_trans)
        self.menu_gray_trans.addAction(self.act_log_trans)
        self.menu_gray_trans.addAction(self.act_hist_eql)
        # -*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-* #

        self.act_sobel = QAction('Sobel算子边缘提取', self.centralwidget)

        # -*-*-*-*-*-**-*     形态学处理菜单     *-*-*-*-*-*-*-* #
        self.menu_morphology = QMenu('形态学处理', self.centralwidget)
        self.act_erode = QAction('腐蚀', self.centralwidget)
        self.act_dilate = QAction('膨胀', self.centralwidget)
        self.act_opening = QAction('开运算', self.centralwidget)
        self.act_closing = QAction('闭运算', self.centralwidget)
        self.menu_morphology.addAction(self.act_erode)
        self.menu_morphology.addAction(self.act_dilate)
        self.menu_morphology.addAction(self.act_opening)
        self.menu_morphology.addAction(self.act_closing)
        # -*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-* #

        # -*-*-*-*-*-**-*     空域滤波菜单     *-*-*-*-*-*-*-* #
        self.menu_filters = QMenu('空域滤波', self.centralwidget)
        self.act_filter_mean = QAction('均值滤波', self.centralwidget)
        self.act_filter_median = QAction('中值滤波', self.centralwidget)
        self.act_filter_gaussian = QAction('高斯滤波', self.centralwidget)
        self.menu_filters.addAction(self.act_filter_mean)
        self.menu_filters.addAction(self.act_filter_median)
        self.menu_filters.addAction(self.act_filter_gaussian)
        # -*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-* #

        self.act_conc = QAction('连通域分析', self.centralwidget)

        # -*-*-*-*-*-**-*     仿射变换菜单     *-*-*-*-*-*-*-* #
        self.menu_affine = QMenu('仿射变换', self.centralwidget)
        self.act_translate = QAction('平移', self.centralwidget)
        self.act_flip = QAction('翻转', self.centralwidget)
        self.act_zoom = QAction('缩放', self.centralwidget)
        self.act_rotate = QAction('旋转', self.centralwidget)
        self.menu_affine.addAction(self.act_translate)
        self.menu_affine.addAction(self.act_flip)
        self.menu_affine.addAction(self.act_zoom)
        self.menu_affine.addAction(self.act_rotate)
        # -*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-* #

        self.act_help = QAction('帮助', self.centralwidget)

        # 将menu和action添加到菜单栏
        self.menu_bar.addMenu(self.menu_file)
        self.menu_bar.addAction(self.act_plot_hist)
        self.menu_bar.addMenu(self.menu_gray_trans)
        self.menu_bar.addAction(self.act_sobel)
        self.menu_bar.addMenu(self.menu_morphology)
        self.menu_bar.addMenu(self.menu_filters)
        self.menu_bar.addAction(self.act_conc)
        self.menu_bar.addMenu(self.menu_affine)
        self.menu_bar.addAction(self.act_help)


        # 用于图像显示的label
        # 要在Qt界面中显示图片，需要一定的容器来装载，有QGraphView、QLabel等多种方法，这里选择QLabel，最为简单
        # 先将图像矩阵转换为QImage对象，再转换为QPixmap，调用QLabel的setPixmap方法，实现图像的显示
        # 在Func.py文件中的show_image方法接收对应的label参数，即可将图像加载到label中显示
        # 预设了三个QLabel，若有更多图片，可在此处添加
        self.label_src = QLabel(self.centralwidget)
        self.label_dst = QLabel(self.centralwidget)
        self.label_ext = QLabel(self.centralwidget)

        self.label_src.move(30, 30)
        self.label_dst.move(30, 30)
        self.label_ext.move(30, 30)

        # 帮助页
        self.widget_help = QMainWindow()
        self.widget_help.setWindowTitle('HELP')
        self.widget_help.setGeometry(40, 40, 960, 700)

        # 设置快捷键
        self.act_open.setShortcut('Ctrl+O')
        self.act_close.setShortcut('Ctrl+C')
        self.act_save.setShortcut('Ctrl+S')
        self.act_quit.setShortcut('Ctrl+Q')
        self.act_plot_hist.setShortcut('Ctrl+P')
        self.act_help.setShortcut('Ctrl+H')
        self.act_prev.setShortcut('Ctrl+up')
        self.act_next.setShortcut('Ctrl+down')

