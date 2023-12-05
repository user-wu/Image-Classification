import random

import requests
import time
from lxml import etree
import re
import os
import urllib
from urllib import request
import matplotlib.pyplot as plt
import PIL.Image as Image

X = 0
ALL = []
# 所模拟的浏览器参数
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1",
    "Accept-Encoding": "gzip, deflate, br"
}

def generate_url(keyword):
    # 将关键词构造为URL
    code_keyword = urllib.parse.quote(keyword, safe='/')
    url = r'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={}'.format(code_keyword)
    return url

# 获取图片
def get_imgs(page_url, keyword):
    data = requests.get(page_url, headers=HEADERS).content.decode("gbk", "ignore")
    imgs_url_list = re.findall('"objURL":"(.*?)",', data, re.S)
    imgs_url_list = [a for a in imgs_url_list if 'jpg' in a]
    # 获取程序的目录的绝对路径
    path1 = os.path.abspath('.')
    # 文件地址和文件名
    global X, ALL
    for i in range(20):
        localfile = path1 + '/Dataset/' + keyword + '/' + str(X) + '.jpg'
        # 判断图片是否在以前爬取过
        if imgs_url_list[i] not in ALL:
            try:
                ALL.append(imgs_url_list[i])
                request.urlretrieve(imgs_url_list[i], localfile)
                # print(random.uniform(0, 1))
                # time.sleep(random.uniform(0, 1))  # Sleep for 1 second
                X += 1
                print("爬取成功{},url={}".format(X, imgs_url_list[i]))
            except Exception as e:
                print(e)
        else:
            print("该图片已存在，爬取下一张")

def get_picture(keyword, num):
    global X
    X = 0
    # 爬取图片的数量
    num = num
    # 关键词
    url = generate_url(keyword)
    # 获取当前文件的绝对路径
    path1 = os.path.abspath('.')

    # 创建Dataset目录
    dataset_path = os.path.join(path1, 'Dataset')
    try:
        os.mkdir(dataset_path)
    except FileExistsError as e:
        print("Dataset文件夹已存在")

    # 创建关键词目录
    keyword_path = os.path.join(dataset_path, keyword)
    try:
        os.mkdir(keyword_path)
    except FileExistsError as e:
        print(f"{keyword}文件夹已存在")

    # 根据关键词生成网页
    url = generate_url(keyword)
    page_num = 0
    # 当爬取的图片数量小于要爬取的数量，循环一直进行
    while X < num:
        page_num = page_num + 20
        page_url = url + '&pn=' + str(page_num)
        print(page_url)
        get_imgs(page_url, keyword)

# 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
# 当你改变搜索类别数时，记得更改后面网络结构，改为对应的分类
keywords = ['猫', '狗']
# 爬取的图片数目
num = 50
for i in range(len(keywords)):
    get_picture(keywords[i], num=num)

# 查看数据图片
# path='猫/1.jpg'
# img = Image.open(path)
# plt.imshow(img)          #根据数组绘制图像
# plt.show()               #显示图像
