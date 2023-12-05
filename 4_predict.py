import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

# 定义图像预处理的transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载模型（假设你的模型保存在 resnet_model.pth）
model = torch.load('out/model.pth')
model.eval()  # 设置模型为评估模式


def predict_image(image_path, model):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    # 进行图像预处理
    input_tensor = transform(image)
    # 添加一个批次维度（batch dimension）
    input_batch = input_tensor.unsqueeze(0)

    # 将图像输入模型进行预测
    with torch.no_grad():
        output = model(input_batch)

    # 获取预测结果
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()


if __name__ == "__main__":
    # 用于预测的图像文件路径（替换为你的图像路径）
    xlsx_path = "./Dataset/class_name.xlsx"
    image_path = './Dataset/000/2.jpg'

    # 进行图像预测
    predicted_class = predict_image(image_path, model)

    xlsx = pd.read_excel(xlsx_path, sheet_name='Sheet1', header=None)

    # 获得文件的行数和列数
    row_num = xlsx.shape[0]
    col_num = xlsx.shape[1]

    name_list = []
    for row in range(row_num):
        temp_list = []
        for col in range(col_num):
            temp_list.append(xlsx.iloc[row, col])
        name_list.append(temp_list)

    print(f'该图像预测的类别为: {name_list[predicted_class+1][1]}')
