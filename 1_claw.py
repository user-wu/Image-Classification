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
import pandas as pd

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
def get_imgs(page_url, category):
    data = requests.get(page_url, headers=HEADERS).content.decode("gbk", "ignore")
    imgs_url_list = re.findall('"objURL":"(.*?)",', data, re.S)
    imgs_url_list = [a for a in imgs_url_list if 'jpg' in a]
    # 获取程序的目录的绝对路径
    path1 = os.path.abspath('.')
    # 文件地址和文件名
    global X, ALL
    category_folder = os.path.join(path1, 'Dataset', category)

    for i in range(20):
        localfile = os.path.join(category_folder, f'{X}.jpg')
        # 判断图片是否在以前爬取过
        if imgs_url_list[i] not in ALL:
            try:
                start_time = time.time()  # Record the start time
                ALL.append(imgs_url_list[i])
                request.urlretrieve(imgs_url_list[i], localfile)
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                if elapsed_time > 2:
                    print(f"等待时间超过两秒，跳过当前图像，url={imgs_url_list[i]}")
                else:
                    X += 1
                    # Calculate the remaining time to wait (2 seconds total waiting time)
                    # remaining_time = max(0, 2 - elapsed_time)
                    # time.sleep(remaining_time)  # Sleep for the remaining time
                    print(f"爬取成功{X}, category={category}, url={imgs_url_list[i]}")
            except Exception as e:
                print(e)
        else:
            print("该图片已存在，爬取下一张")

# 生成Excel文件
def generate_excel():
    data = {'Image': [], 'Category': []}
    for category in ['001', '002']:
        category_folder = f'Dataset/{category}'
        for filename in os.listdir(category_folder):
            if filename.endswith('.jpg'):
                data['Image'].append(filename)
                data['Category'].append(category)

    df = pd.DataFrame(data)
    df.to_excel('class_name.xlsx', index=False)

# 获取图片
def get_picture(keyword, num, category):
    global X
    X = 0
    # 爬取图片的数量
    num = num
    # 获取当前文件的绝对路径
    path1 = os.path.abspath('.')
    # 创建文件夹
    category_folder = os.path.join(path1, 'Dataset', category)
    try:
        os.makedirs(category_folder, exist_ok=True)
    except FileExistsError as e:
        print("文件夹已存在")
    # 根据关键词生成网页
    url = generate_url(keyword)
    page_num = 0
    # 当爬取的图片数量小于要爬取的数量，循环一直进行
    while X < num:
        page_num = page_num + 2
        page_url = url + '&pn=' + str(page_num)
        print(page_url)
        get_imgs(page_url, category)

# 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
# 当你改变搜索类别数时，记得更改后面网络结构，改为对应的分类
def main():
    keywords = ['猫', '狗',"大象"]
    num = 50
    label = [str(i).zfill(3) for i in range(len(keywords))]

    for i, category in enumerate(label):
        get_picture(keywords[i], num=num, category=category)

    data = {'Label': [], 'Category': []}
    for l in label:
        data['Label'].append(l)
        data['Category'].append(keywords[label.index(l)])

    df = pd.DataFrame(data)
    df.to_excel('./Dataset/class_name.xlsx', index=False)

if __name__ == "__main__":
    main()
