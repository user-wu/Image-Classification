import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
from tqdm import tqdm  # 导入tqdm



# 1. 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx].split()  # 假设文件列表的每一行是 "path/to/image.jpg label"
        image = Image.open("./SplitDataset/"+img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, int(label)


# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 加载数据集
train_file_list = [line.strip() for line in open('./SplitDataset/train_list.txt', 'r')]
val_file_list = [line.strip() for line in open('./SplitDataset/val_list.txt', 'r')]
test_file_list = [line.strip() for line in open('./SplitDataset/test_list.txt', 'r')]

train_dataset = CustomDataset(file_list=train_file_list, transform=transform)
val_dataset = CustomDataset(file_list=val_file_list, transform=transform)
test_dataset = CustomDataset(file_list=test_file_list, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

# 4. 定义 ResNet 模型
model = torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 假设 num_classes 是你的分类类别数

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 设置 TensorBoard
writer = SummaryWriter()

# 6. 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs, labels, "<==output")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # 计算平均损失和准确率
    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    # 将训练过程中的信息写入 TensorBoard
    writer.add_scalar('Train/Loss', average_loss, epoch + 1)
    writer.add_scalar('Train/Accuracy', accuracy, epoch + 1)

    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')


    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')
# 关闭 TensorBoard writer
writer.close()
# 假设你的模型命名为model
# 保存整个模型，包括模型结构和参数
head = "./out/"
if not os.path.exists(head):
    os.makedirs(head)
torch.save(model, 'out/model.pth')

# 在测试集上测试模型
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
