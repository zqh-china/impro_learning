import copy
import numpy as np
from PyQt5.QtGui import QPixmap, qRgb, QImage
from PIL import Image
import struct

# 结构元素
MORPH_CROSS = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], int)

MORPH_RECT = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], int)


def open_image(path):
    dst = np.zeros((400, 300), np.uint8)
    if path.endswith('.raw'):
        dst = read_raw(path)
    elif path.endswith('.bmp') or path.endswith('.BMP'):
        dst = read_bmp(path)
    return dst


# def read_raw(path):
#     w, h = np.fromfile(path, dtype=np.uint32)[:2]
#     dst = np.fromfile(path, dtype=np.uint8)[8:]
#     dst = dst.reshape(h, w)
#     return dst


def read_raw(path):
    with open(path, 'rb') as f:
        data = f.read()
    # data是一个python中的一种特殊的字符串，叫做“字节流”，每一个单元就是一个字节
    w_byte = data[:4]  # 前四个字节是图片的宽 读取出来是 b'\x80\x01\x00\x00' 等于 0*1+8*16+1*256 = 384
    h_byte = data[4:8]  # 第5-8字节是图片的高
    im = data[8:]  # 第8个字节以后才是图像的像素灰度值，一个灰度值刚好是一个字节
    w = struct.unpack('I', w_byte)[0]  # 将四个字节的字节流转换为无符号整型
    h = struct.unpack('I', h_byte)[0]
    dst = np.zeros((h, w), dtype=np.uint8)  # 创建一个h x w 的全零矩阵dst，所有元素都是8位无符号整型，也就是0-255

    for i in range(h):
        for j in range(w):
            dst[i][j] = im[i*w + j]  # 将im中的灰度值逐个存入dst
    return dst


# 读取bmp格式，关于位图的读取，可参考https://blog.csdn.net/qq_43409114/article/details/104538619，这里直接用PIL了
def read_bmp(path):
    im = Image.open(path)
    dst = np.array(im)
    return dst


# 显示图像，image为图像像素灰度的矩阵，label为用于装载图片的组件，x, y为label在界面的位置
def show_image(image, label, x, y):
    h, w = image.shape[:2]
    qimg = QImage(w, h, QImage.Format_RGB32)
    # 彩色图像
    if image.ndim == 3:
        for i in range(h):
            for j in range(w):
                val = qRgb(image[i][j][0], image[i][j][1], image[i][j][2])  # rgb
                qimg.setPixel(j, i, val)
    # 灰度图像
    elif image.ndim == 2:
        for i in range(h):
            for j in range(w):
                val = qRgb(image[i][j], image[i][j], image[i][j])
                qimg.setPixel(j, i, val)
    pixmap = QPixmap.fromImage(qimg)
    label.setPixmap(pixmap)
    label.setGeometry(x, y, w, h)
    label.show()
    label.raise_()  # 置于最上层


# 计算直方图
def cal_hist(src):
    hist = np.zeros(256, dtype=int)  # 先赋值一个全零数组，用来统计各个灰度级像素个数
    h, w = src.shape[:2]  # 取图像矩阵的高和宽
    for i in range(h):
        for j in range(w):
            hist[src[i][j]] += 1  # 统计
    return hist


# 绘制直方图的图像
def plot_hist(src):
    hist = cal_hist(src)
    # 直方图单独看作一张图片，设置其高度为400px，宽度为1024px，即为256的4倍，这样能够显示出一条条分立的谱线
    img_h = 400
    img_w = 256 * 4
    max_val = hist.max()
    if max_val == 0:
        max_val = 1000000
    coeffs = (400 - 20) / max_val  # 直方图归一化，减20是为了不到顶
    normed_hist = hist * coeffs
    normed_hist = normed_hist.astype(int)

    dst = np.ones((img_h, img_w), dtype=np.uint8) + 239  # 用于保存直方图的图片的矩阵
    for i in range(img_w >> 2):  # 右移两位，相当于除以4
        for j in range(img_h, 0, -1):
            if j < normed_hist[i]:
                dst[400 - j][i << 2] = 0
    return dst


# 二值化
def binarize(src, th):
    h, w = src.shape[:2]
    dst = np.zeros_like(src)
    for i in range(h):
        for j in range(w):
            if src[i][j] > th:
                dst[i][j] = 255
    return dst


# # 幂次变换 如果对python和numpy熟悉，可以使用此版本
# def exp_trans(src, expon):
#     data = src.astype(np.float64)  # 先转换数据类型
#     tmp = np.power(data, expon)  # numpy数组的广播特性，会将运算广播至每一个元素
#     coeffs = 255 / tmp.max()
#     tmp = tmp * coeffs
#     dst = tmp.astype(np.uint8)
#     return dst


# 幂次变换 细节版
def exp_trans(src, expon):
    dst = np.zeros_like(src)
    data = src.astype(np.float64)  # 由于幂次变换的数值较大，我们先取一中间变量，设其数据类型为float64
    h, w = src.shape
    for i in range(h):
        for j in range(w):
            data[i][j] = np.power(data[i][j], expon)
    coeffs = 255 / data.max()
    for i in range(h):
        for j in range(w):
            dst[i][j] =  coeffs * data[i][j]
    return dst


# 幂次变换-3通道
def exp_trans_bmp(src, expon):
    h, w = src.shape[:2]
    src_r = src[:, :, 0]
    src_g = src[:, :, 1]
    src_b = src[:, :, 2]
    dst_r = exp_trans(src_r, expon)
    dst_g = exp_trans(src_g, expon)
    dst_b = exp_trans(src_b, expon)
    dst = np.zeros((h, w, 3))
    dst[:, :, 0] = dst_r
    dst[:, :, 1] = dst_g
    dst[:, :, 2] = dst_b
    return dst


# 对数变换
def log_trans(src):
    dst = np.zeros_like(src)
    data = src.astype(np.float16)  # 先转换数据类型
    tmp = np.log(data + 1)
    coeffs = 255 / tmp.max()
    tmp = tmp * coeffs
    dst = tmp.astype(np.uint8)
    return dst


# 对数变换-3通道
def log_trans_bmp(src):
    h, w = src.shape[:2]
    src_r = src[:, :, 0]
    src_g = src[:, :, 1]
    src_b = src[:, :, 2]
    dst_r = log_trans(src_r)
    dst_g = log_trans(src_g)
    dst_b = log_trans(src_b)
    dst = np.zeros((h, w, 3))
    dst[:, :, 0] = dst_r
    dst[:, :, 1] = dst_g
    dst[:, :, 2] = dst_b
    return dst


# 直方图均衡
def hist_eql(src):
    h, w = src.shape[:2]
    # 计算直方图
    hist = cal_hist(src)
    # 求概率密度直方图（未归一化）
    hist_pdf = hist.astype(np.float32)
    # 求累积分布直方图
    hist_cdf = np.zeros(256, dtype=np.float32)
    hist_cdf[0] = hist_pdf[0]
    for i in range(1, 256):
        hist_cdf[i] = hist_pdf[i] + hist_cdf[i - 1]
    coeffs = 255 / (w*h)
    hist_map = np.zeros(256, np.uint8)
    for i in range(256):
        hist_map[i] = coeffs * hist_cdf[i]
    dst = np.zeros_like(src)
    for i in range(h):
        for j in range(w):
            dst[i][j] = hist_map[src[i][j]]
    return dst


# sobel算子边缘提取
def sobel(src):
    sob_x = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    sob_y = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])

    h, w = src.shape[:2]
    # 构造与src同形状的零矩阵
    sobel_x = np.zeros_like(src)
    sobel_y = np.zeros_like(src)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            block = src[i-1:i+2, j-1:j+2]  # 切片操作，取(i, j)为中心，3 x 3的一个小方框（8邻域）
            sobel_x[i][j] = np.abs((sob_x * block).sum())
            sobel_y[i][j] = np.abs((sob_y * block).sum())
    # sobel_x = tranByte(sobel_x)
    # sobel_y = tranByte(sobel_y)
    return sobel_x, sobel_y


# # 腐蚀
# def erode(src, iterations=1):
#     h, w = src.shape[:2]
#     dst = copy.deepcopy(src)
#     tmp = copy.deepcopy(src)
#     for k in range(iterations):
#         for i in range(1, h - 1):
#             for j in range(1, w - 1):
#                 if tmp[i][j] == 255:  # 对图像点操作
#                     mask = dst[i - 1:i + 2, j - 1:j + 2].reshape(-1)  # 取(i, j)为中心，3 x 3的一个小方框（8邻域）
#                     z_count = 0
#                     for m in range(9):
#                         if mask[m] == 0:
#                             z_count += 1
#                         if z_count > 5:
#                             tmp[i][j] = 0
#                             break
#         # 之所以要用一个tmp来把值拷贝给dst，是因为要保持每次迭代的基础图像不变，否则，从上到下从左到右的腐蚀下来，上一行腐蚀的结果会影响下一行
#         dst = copy.deepcopy(tmp)
#     return dst


# 腐蚀
def erode(src, iterations=1):
    h, w = src.shape[:2]
    dst = copy.deepcopy(src)
    tmp = copy.deepcopy(src)
    for k in range(iterations):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if tmp[i][j] == 255:  # 对图像点操作
                    mask = MORPH_CROSS
                    block = dst[i-1:i+2, j-1:j+2]
                    val = (block * mask).sum()
                    if val < 255 * mask.sum():
                        tmp[i][j] = 0
        # 之所以要用一个tmp来把值拷贝给dst，是因为要保持每次迭代的基础图像不变，否则，从上到下从左到右的腐蚀下来，上一行腐蚀的结果会影响下一行
        dst = copy.deepcopy(tmp)
    return dst


# 膨胀
def dilate(src, iterations=1):
    h, w = src.shape[:2]
    dst = copy.deepcopy(src)
    tmp = copy.deepcopy(src)
    for k in range(iterations):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if tmp[i][j] == 0:  # 对背景点操作
                    mask = MORPH_CROSS
                    block = dst[i-1:i+2, j-1:j+2]
                    val = (block * mask).sum()
                    if val >= 255:
                        tmp[i][j] = 255
        dst = copy.deepcopy(tmp)
    return dst


# 开运算
def opening(src, iterations):
    dst = copy.deepcopy(src)
    for i in range(iterations):
        ero_img = erode(dst, 1)
        dst = dilate(ero_img, 1)
    return dst


# 闭运算
def closing(src, iterations):
    dst = copy.deepcopy(src)
    for i in range(iterations):
        dil_img = dilate(dst, 1)
        dst = erode(dil_img, 1)
    return dst


# # 均值滤波 不考虑图像最外的一圈像素
# def mean_filter(src):
#     h, w = src.shape[:2]
#     dst = np.zeros((h, w), dtype=np.uint8)
#     for i in range(1, h-1):
#         for j in range(1, w-1):
#             block = src[i-1:i+2, j-1:j+2]
#             dst[i][j] = int(block.mean())  # numpy数组的方法mean，返回平均值
#     return dst


# 均值滤波
def mean_filter(src):
    h, w = src.shape[:2]

    # 将待处理的基础图像扩展一圈，使得边缘处的像素点也可以被遍历到
    tmp = np.zeros((h + 2, w + 2), dtype=np.uint8)
    dst = np.zeros((h, w), dtype=np.uint8)
    tmp[1:-1, 1:-1] = copy.deepcopy(src)
    tmp[:, -1] = tmp[:, 1]
    tmp[:, 0] = tmp[:, -2]
    tmp[-1, :] = tmp[1, :]
    tmp[0, :] = tmp[-2, :]

    for i in range(h):
        for j in range(w):
            block = src[i:i + 3, j:j + 3]
            dst[i][j] = int(block.mean())  # numpy数组的方法mean，返回平均值
    return dst


# 中值滤波
def median_filter(src):
    h, w = src.shape[:2]

    # 将待处理的基础图像扩展一圈，使得边缘处的像素点也可以被遍历到
    tmp = np.zeros((h + 2, w + 2), dtype=np.uint8)  # 在原图像的基础上构造一个比它大一圈的图像
    dst = np.zeros((h, w), dtype=np.uint8)
    tmp[1:-1, 1:-1] = copy.deepcopy(src)  # 这个大一圈的矩阵除去边缘之外的值赋值为原图像
    tmp[:, -1] = tmp[:, 1]  # 用原图像的第1行(tmp的第2行)补到tmp的最后一行
    tmp[:, 0] = tmp[:, -2]  # 用原图像的最后1行(tmp的倒数第2行)补到tmp的第1行
    tmp[-1, :] = tmp[1, :]
    tmp[0, :] = tmp[-2, :]

    for i in range(h):
        for j in range(w):
            block = tmp[i:i + 3, j:j + 3].reshape(-1)
            block.sort()  # numpy的sort方法，排序
            dst[i][j] = block[len(block) // 2]  # 返回中位数
    return dst


# 高斯滤波
def gaussian_filter(src):
    h, w = src.shape[:2]
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])

    # 将待处理的基础图像扩展一圈，使得边缘处的像素点也可以被遍历到
    tmp = np.zeros((h + 2, w + 2), dtype=np.uint8)
    dst = np.zeros((h, w), dtype=np.uint8)
    tmp[1:-1, 1:-1] = copy.deepcopy(src)
    tmp[:, -1] = tmp[:, 1]
    tmp[:, 0] = tmp[:, -2]
    tmp[-1, :] = tmp[1, :]
    tmp[0, :] = tmp[-2, :]

    for i in range(h):
        for j in range(w):
            block = tmp[i:i + 3, j:j + 3] * kernel
            dst[i][j] = block.sum() >> 4  # 右移4位即除以16，移位运算比浮点数除法快
    return dst


# 连通域分析
def conc(src):
    flag = 0
    h, w = src.shape[:2]
    labels = np.zeros((h, w))
    # 粗检测 顺序检测 检测对象是src
    for i in range(1, h-1):
        for j in range(1, w-1):
            if src[i][j] == 255:  # 只对src中的图形点进行检测
                block = labels[i-1:i+2, j-1:j+2]  # 一个像素的8邻域
                neighbors = []  # 连通域
                for m in range(3):
                    for n in range(3):
                        if block[m][n] > 0:  # 只取邻域中的标记
                            neighbors.append(block[m][n])
                if not neighbors:  # 列表为空，说明周围都没有标记，则开始一个新的连通域标记
                    flag += 1
                    labels[i][j] = flag
                else:
                    labels[i][j] = min(neighbors)
    # 复查 逆序检测 检测对象是labels
    for i in range(h-2, 0, -1):
        for j in range(w-2, 0, -1):
            if labels[i][j] > 0:  # 只对labels中标记的连通域进行检测
                block = labels[i-1:i+2, j-1:j+2]
                neighbors = []
                for m in range(3):
                    for n in range(3):
                        if block[m][n] > 0:
                            neighbors.append(block[m][n])
                labels[i][j] = min(neighbors)
    # 重新映射灰度级
    step = 255 // flag  # 此时的flag已经是labels中的最大值
    labels = labels * step
    labels = labels.astype(np.uint8)
    return labels


# 平移变换
def translate(src, delta_x, delta_y):
    h, w = src.shape[:2]
    # 若平移的偏移量大于图像的宽高，则取其对应的模宽或高的运算结果
    if delta_x >= w or delta_x <= -w:
        delta_x = delta_x % w
    if delta_y >= h or delta_y <= -h:
        delta_y = delta_y % h
    dst = np.zeros_like(src)
    M = np.array([[1, 0, delta_x],
                  [0, 1, delta_y]])
    for i in range(h):
        for j in range(w):
            position = np.array([[j],
                                 [i],
                                 [1]])

            new_position = M @ position  # 矩阵相乘，得到变换后的位置
            x, y = new_position[:, 0]
            # 移除图像外的点不管
            if x >= w:
                continue
                # x = x % w
            if y >= h:
                continue
                # y = y % h
            dst[y][x] = src[i][j]
    return dst


# 翻转变换
def flip(src, M):
    h, w = src.shape[:2]
    dst = np.zeros_like(src)
    for i in range(h):
        for j in range(w):
            position = np.array([[j],
                                 [i],
                                 [1]])

            new_position = M @ position  # 矩阵相乘，得到变换后的位置
            x, y = new_position[:, 0]
            dst[y][x] = src[i][j]
    return dst


# 缩放
def zoom(src, f_x, f_y):
    h, w = src.shape[:2]
    new_h = int(h * f_y)
    new_w = int(w * f_x)
    dst = np.zeros((new_h, new_w), dtype=np.uint8)

    M = np.array([[f_x, 0, 0],
                  [0, f_y, 0]])
    for i in range(h):
        for j in range(w):
            position = np.array([[j],
                                 [i],
                                 [1]])

            new_position = M @ position
            x, y = new_position[:, 0]
            x = int(x)
            y = int(y)
            dst[y][x] = src[i][j]
    tmp = np.zeros((new_h + 2, new_w + 2), dtype=np.uint8)
    tmp[1:-1, 1:-1] = copy.deepcopy(dst)
    tmp[:, -1] = tmp[:, 1]
    tmp[:, 0] = tmp[:, -2]
    tmp[-1, :] = tmp[1, :]
    tmp[0, :] = tmp[-2, :]
    for i in range(1, new_h + 1):
        for j in range(1, new_w + 1):
            if tmp[i][j] == 0:
                block = tmp[i - 1:i + 2, j - 1:j + 2]  # 取(i,j)为中心的3x3的一个小块
                new_val = mean_value(block)
                dst[i - 1][j - 1] = new_val
    return dst


# 旋转
def rotate(src, theta):
    theta = theta / 180 * np.pi  # 角度转弧度

    h, w = src.shape[:2]
    cx, cy = w >> 1, h >> 1  # 图片中心
    M = np.array([[+np.cos(theta), +np.sin(theta), (1 - np.cos(theta) * cx - cy * np.sin(theta))],
                  [-np.sin(theta), +np.cos(theta), (1 - np.cos(theta) * cy + cx * np.sin(theta))]])
    new_h = int(w * np.sin(theta) + h * np.cos(theta))
    new_w = int(h * np.sin(theta) + w * np.cos(theta))
    dst = np.zeros((new_h, new_w), dtype=np.uint8)
    new_cx, new_cy = new_w >> 1, new_h >> 1

    for i in range(h):
        for j in range(w):
            position = np.array([[j],
                                 [i],
                                 [1]])

            new_position = M @ position
            x, y = new_position[:, 0]
            x = int(x + new_cx)
            y = int(y + new_cy)  # 移到中心
            if x < 0:
                continue
            if y < 0:
                continue
            if x >= new_w:
                x -= new_w
            if y >= new_h:
                y -= new_h
            dst[y][x] = src[i][j]
    # 当倾斜角度不是90的整数倍时会产生缝隙，用一个像素周围非零像素的均值来填充这些缝隙
    tmp = np.zeros((new_h+2, new_w+2), dtype=np.uint8)
    tmp[1:-1, 1:-1] = copy.deepcopy(dst)
    tmp[:, -1] = tmp[:, 1]
    tmp[:, 0] = tmp[:, -2]
    tmp[-1, :] = tmp[1, :]
    tmp[0, :] = tmp[-2, :]
    for i in range(1, new_h+1):
        for j in range(1, new_w+1):
            if tmp[i][j] == 0:
                block = tmp[i-1:i+2, j-1:j+2]  # 取(i,j)为中心的3x3的一个小块
                new_val = mean_value(block)
                dst[i-1][j-1] = new_val
    return dst


# 3x3非零均值填充
def mean_value(block):
    seq = block.reshape(-1)
    count = 0
    sigma = 0
    for i in range(9):
        if seq[i] != 0:
            count += 1
            sigma += seq[i]
    if count == 0:
        count = 1
    return int(sigma / count)


def tranByte(array):
    dim = array.size
    h, w = array.shape[:2]
    mean = array.mean()
    stm = 120
    stvar = 60
    var = array.var()
    if var < 0.00001:
        return array
    ratio = stvar / np.sqrt(var)
    array = array.reshape(-1).tolist()
    for i in range(dim):
        value = np.uint8((array[i] - mean) * ratio + stm)
        if value >= 200:
            array[i] = 200
        elif value < 80:
            array[i] = 80
        else:
            array[i] = value
    array = np.array(array, dtype=np.uint8).reshape(w, h)
    return array
