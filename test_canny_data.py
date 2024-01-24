import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

#遍历文件夹
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
 
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist

org_img_folder = '/data/rensisi/ControlNet-main/new_data/skirt'
imglist = getFileList(org_img_folder, [], 'jpg')
# print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')

for imgpath in imglist:
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
        #print(imgname)
        #print("dataset1/shiban1/cut/" + imgname + ".png")
    imgpath = "/data/rensisi/ControlNet-main/new_data/skirt/" + imgname + ".jpg"
    # print(imgpath)
    img = cv2.imread(imgpath)
    # print(img)
    edges = cv2.Canny(img,100,200)
    cv2.imwrite('/data/rensisi/ControlNet-main/new_data/skirt_canny/{}.jpg'.format(imgname), edges)
    print('{} done'.format(imgname))

