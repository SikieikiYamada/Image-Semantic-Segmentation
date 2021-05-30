import numpy as np
# from PIL import Image
import cv2

in_path = './images/'
out_path = './test_res/'

for file in range(2100):
    print('running '+str(file))
    # 读取图片，保存长宽后将图片格式化
    img = cv2.imread(in_path+str(file)+'.jpg')
    height, width = img.shape[:2]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 为俩个数组分配内存，GrabCut算法从背景中分割前景时会用到
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 使用GrabCut应用边界框分割方法
    rect = (5, 5, 224, 224)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # 将背景设为0，图片放缩回原来的尺寸
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # 保存图片
    cv2.imwrite(out_path + str(file) + '.png', img)
