import os
import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

from deps.UI_MainWindow import MainWindow
from deps.Func import *
from deps.UI_PopWindow import PopWidget
MAIN_PATH = os.path.dirname(os.path.abspath(__file__))

class Window:
    def __init__(self):
        self.ui = MainWindow()
        # self.pop_window = PopWidget()
        self.image = None  # 类的属性作为`全局变量`
        self.images = []
        self.index = -1
        self.initUi()

    def initUi(self):
        # 绑定信号(Signal)与槽(Slot)，当菜单栏对应位置被点击时，会触发对应的槽函数来响应
        self.ui.act_open.triggered.connect(self.slot_open)
        self.ui.act_open_folder.triggered.connect(self.slot_open_folder)
        self.ui.act_close.triggered.connect(self.slot_close)
        self.ui.act_save.triggered.connect(self.slot_save)
        self.ui.act_quit.triggered.connect(self.slot_quit)
        self.ui.act_prev.triggered.connect(self.slot_prev)
        self.ui.act_next.triggered.connect(self.slot_next)

        self.ui.act_plot_hist.triggered.connect(self.slot_plot_hist)

        self.ui.act_binarize.triggered.connect(self.slot_binarize)
        self.ui.act_exp_trans.triggered.connect(self.slot_exp_trans)
        self.ui.act_log_trans.triggered.connect(self.slot_log_trans)
        self.ui.act_hist_eql.triggered.connect(self.slot_hist_eql)

        self.ui.act_sobel.triggered.connect(self.slot_sobel)

        self.ui.act_erode.triggered.connect(self.slot_erode)
        self.ui.act_dilate.triggered.connect(self.slot_dilate)
        self.ui.act_opening.triggered.connect(self.slot_opening)
        self.ui.act_closing.triggered.connect(self.slot_closing)

        self.ui.act_filter_mean.triggered.connect(self.slot_filter_mean)
        self.ui.act_filter_median.triggered.connect(self.slot_filter_median)
        self.ui.act_filter_gaussian.triggered.connect(self.slot_filter_gaussian)

        self.ui.act_conc.triggered.connect(self.slot_conc)

        self.ui.act_translate.triggered.connect(self.slot_translate)
        self.ui.act_flip.triggered.connect(self.slot_flip)
        self.ui.act_zoom.triggered.connect(self.slot_zoom)
        self.ui.act_rotate.triggered.connect(self.slot_rotate)

        self.ui.act_help.triggered.connect(self.slot_help)

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #
    # 以下都是槽函数，在initUi中绑定了信号与槽函数之后，相应的动作会得到对应的槽函数的响应

    # 打开图片
    def slot_open(self):
        # 加载设置，保存了上一次打开图片的路径
        setting = QSettings(os.path.join(MAIN_PATH, "config/setting.ini"), QSettings.IniFormat)
        last_path = setting.value("LastFilePath")
        if last_path is None:
            last_path = './'
        file_name = QFileDialog.getOpenFileName(self.ui, "打开文件", last_path, "*.bmp *.raw")[0]
        # 打开文件对话框后取消的情况
        if file_name is None or file_name == '':
            return
        setting.setValue("LastFilePath", file_name)
        # 在打开操作前将打开目录产生的数据清零
        self.index = -1
        self.images = []

        self.image = open_image(file_name)
        self.clear_figure()
        show_image(self.image, self.ui.label_src, 20, 60)

    # 打开目录
    def slot_open_folder(self):
        setting = QSettings(os.path.join(MAIN_PATH, "config/setting.ini"), QSettings.IniFormat)
        last_dir = setting.value("LastDir")
        if last_dir is None:
            last_dir = './'
        dir_name = QFileDialog.getExistingDirectory(self.ui, "打开目录", last_dir)
        if dir_name is None or dir_name == '':
            return
        setting.setValue("LastDir", dir_name)
        file_names = os.listdir(dir_name)  # 以列表形式返回该目录下的所有文件名
        img_files = []
        # 只保留.raw和.bmp格式的文件名
        for file_name in file_names:
            if file_name.endswith('.raw') or file_name.endswith('.bmp'):
                img_files.append(os.path.join(dir_name, file_name))  # 文件名加上其所在目录，构成完整路径
        self.index = 1  # 当前图片的索引
        self.images = copy.deepcopy(img_files)  # 深拷贝
        self.image = open_image(self.images[self.index])  # 打开当前图片
        self.clear_figure()
        show_image(self.image, self.ui.label_src, 20, 60)  # 显示当前图片

    # 上一张
    def slot_prev(self):
        if self.index <= 1:  # 未打开目录或者没有上一张
            return
        self.index -= 1  # 当前索引 -1
        self.image = self.images[self.index]  # 取图片文件路径
        self.image = open_image(self.images[self.index])  # 打开当前图片
        self.clear_figure()
        show_image(self.image, self.ui.label_src, 20, 60)  # 显示当前图片

    # 下一张
    def slot_next(self):
        if self.index >= len(self.images)-1:  # 没有下一张
            return
        self.index += 1  # 当前索引 +1
        self.image = self.images[self.index]  # 取图片文件路径
        self.image = open_image(self.images[self.index])  # 打开当前图片
        self.clear_figure()
        show_image(self.image, self.ui.label_src, 20, 60)  # 显示当前图片

    # 关闭所有图片
    def slot_close(self):
        self.clear_figure()
        self.index = -1
        self.images = []
        self.image = None

    # 保存图片
    def slot_save(self):
        im = self.ui.label_dst.pixmap()
        if im is None:
            return
        qim = im.toImage()
        w = qim.width()
        h = qim.height()
        dst = np.zeros((h, w), np.uint8)
        for i in range(h):
            for j in range(w):
                pix = qRed(qim.pixel(j, i))
                dst[i][j] = pix
        file_name = QFileDialog.getSaveFileName(self.ui, '保存图片', './', '*.raw')[0]
        # dst.tofile(file_name)
        with open(file_name, 'wb') as f:
            head = struct.pack('ii', w, h)
            for i in range(h):
                for j in range(w):
                    if i == 0 and j == 0:
                        s = struct.pack('B', dst[i][j])
                    else:
                        s += struct.pack('B', dst[i][j])
            s_bytes = head + s
            f.write(s_bytes)

    # 退出应用程序
    def slot_quit(self):
        self.ui.close()

    # 绘制直方图
    def slot_plot_hist(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = plot_hist(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)  # 在显示之前先清空label_dst和label_ext
        show_image(out_image, self.ui.label_dst, 20, h+10)
        self.pop_window.show()

    # 图像二值化
    def slot_binarize(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = binarize(self.image, 127)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 幂次变换
    def slot_exp_trans(self):
        if self.image is None or self.image.ndim == 3:
            return
        gamma = 3.0
        out_image = exp_trans(self.image, gamma)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)
        # 显示gamma
        self.ui.label_ext.setText(str(gamma))  # 设置文本内容
        self.ui.label_ext.setGeometry(w+30, 60, 30, 20)  # 设置文本大小和位置
        self.ui.label_ext.raise_()  # 置于图层最上面
        self.ui.label_ext.setStyleSheet('QLabel {\
                                        background-color: white;\
                                        }')  # 设置文字背景

    # 对数变换
    def slot_log_trans(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = log_trans(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 直方图均衡
    def slot_hist_eql(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = hist_eql(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # Sobel算子边缘提取
    def slot_sobel(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image_x, out_image_y = sobel(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image_x, self.ui.label_dst, w+30, 60)
        show_image(out_image_y, self.ui.label_ext, w*2+40, 60)

    # 腐蚀
    def slot_erode(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = erode(self.image, 1)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 膨胀
    def slot_dilate(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = dilate(self.image, 3)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 开运算
    def slot_opening(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = opening(self.image, 2)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 闭运算
    def slot_closing(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = closing(self.image, 2)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 均值滤波
    def slot_filter_mean(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = mean_filter(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 中值滤波
    def slot_filter_median(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = median_filter(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 高斯滤波
    def slot_filter_gaussian(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = gaussian_filter(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 连通域分析
    def slot_conc(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = conc(self.image)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w + 30, 60)

    # 平移
    def slot_translate(self):
        if self.image is None or self.image.ndim == 3:
            return
        out_image = translate(self.image, 150, 0)
        h, w = self.image.shape[:2]
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 翻转
    def slot_flip(self):
        if self.image is None or self.image.ndim == 3:
            return
        h, w = self.image.shape[:2]
        # 水平翻转
        M_x = np.array([[-1, 0, w-1],
                      [0, 1, 0]])
        # 垂直翻转
        M_y = np.array([[1, 0, 0],
                      [0, -1, h-1]])
        # 既水平也垂直翻转（相当于旋转180°）
        M_xy = np.array([[-1, 0, w-1],
                      [0, -1, h-1]])
        M = M_xy
        out_image = flip(self.image, M)

        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 缩放
    def slot_zoom(self):
        if self.image is None or self.image.ndim == 3:
            return
        h, w = self.image.shape[:2]
        f_x = 2
        f_y = 2
        out_image = zoom(self.image, f_x, f_y)
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 旋转
    def slot_rotate(self):
        if self.image is None or self.image.ndim == 3:
            return
        h, w = self.image.shape[:2]
        theta = 45
        out_image = rotate(self.image, theta)
        self.clear_figure(3)
        show_image(out_image, self.ui.label_dst, w+30, 60)

    # 打开帮助
    def slot_help(self):
        # import os
        browser = QWebEngineView()
        # path = 'README.html'
        # url = os.path.join(os.getcwd(), path)
        # browser.load(QUrl.fromLocalFile(url))
        browser.setUrl('https://gitee.com/zhang_qi_hao/im-pro/blob/master/README.md')
        self.ui.widget_help.setCentralWidget(browser)
        self.ui.widget_help.show()
        self.ui.widget_help.raise_()

    # 清空图片，由于图片都需要一个QLabel去装载，为了方便装载后清理图片，这里预设了3个用于装载图片的QLabel，code决定了清楚哪几张图片
    def clear_figure(self, code=7):
        if code == 7:
            self.ui.label_src.clear()  # 4
            self.ui.label_dst.clear()  # 2
            self.ui.label_ext.clear()  # 1
        elif code == 6:
            self.ui.label_src.clear()  # 4
            self.ui.label_dst.clear()  # 2
        elif code == 5:
            self.ui.label_src.clear()  # 4
            self.ui.label_ext.clear()  # 1
        elif code == 4:
            self.ui.label_src.clear()  # 4
        elif code == 3:
            self.ui.label_dst.clear()  # 2
            self.ui.label_ext.clear()  # 1
        elif code == 2:
            self.ui.label_dst.clear()  # 2
        elif code == 1:
            self.ui.label_ext.clear()  # 1
        else:
            return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.ui.show()
    sys.exit(app.exec_())
