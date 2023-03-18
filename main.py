import csv

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os

use_gpu = torch.cuda.is_available()

buf_size = 1
train_file = open('train.csv', 'w', buffering=buf_size)
train_writer = csv.writer(train_file)
train_writer.writerow(['epoch', 'loss', 'precision', 'recall'])

val_file = open('valid.csv', 'w', buffering=buf_size)
val_writer = csv.writer(val_file)
val_writer.writerow(['epoch', 'loss', 'precision', 'recall'])

# 定义数据的处理方式
data_transforms = {
    'train': transforms.Compose([
        # 将图像缩放为256*256
        transforms.Resize(256),
        # 随机裁剪出227*227大小的图像用于训练
        transforms.RandomResizedCrop(227),
        # 将图像进行水平翻转
        transforms.RandomHorizontalFlip(),
        # 转换为张量
        transforms.ToTensor(),
        # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试样本进行中心裁剪，且无需翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# 加载图像，path是图像的路径
def load_image(image_path):
    # 以RGB格式打开图像
    # Pytorch DataLoader使用PIL的Image格式
    # 需要读取灰度图像时使用 .convert('')
    return Image.open(image_path).convert('RGB')


# 定义自己的数据集读取类
class my_dataset(nn.Module):
    def __init__(self, label_path, images_dir, transform=None, target_transform=None, loader=None):
        super(my_dataset, self).__init__()
        # 打开存储图像名称与标签的txt文件
        fp = open(label_path, 'r')
        images = []
        labels = []
        for line in fp:
            # 移除首位的回车符
            line.strip('\n')
            # 移除末尾的空格符
            line.rstrip()
            info = line.split(',')
            images.append(info[0])
            # 将标签信息转为float类型
            labels.append([float(l) for l in info[1:len(info)]])
        self.images = images
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # 重写该函数进行图像数据的读取
    # 通过索引的访问方式
    def __getitem__(self, item):
        image_name = self.images[item]
        # 这里的label是一个list列表，表示标签向量
        label = self.labels[item]
        # 从loader中根据图像名称读取图像信息
        image = self.loader(os.path.join(self.images_dir, image_name))
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        # 将浮点数的list转换为float tensor
        label = torch.FloatTensor(label)
        # 返回处理后的内容
        return image, label

    def __len__(self):
        return len(self.images)


train_label_path = '../dataset/ms-coco/train2014/labels.txt'
train_images_dir = '../dataset/ms-coco/train2014'
val_label_path = '../dataset/ms-coco/val2014/labels.txt'
val_image_dir = '../dataset/ms-coco/val2014'

train_data = my_dataset(train_label_path, images_dir=train_images_dir, transform=data_transforms['train'],
                        loader=load_image)  # 读取一张图片
val_data = my_dataset(val_label_path, images_dir=val_image_dir, transform=data_transforms['val'], loader=load_image)

batch_size = 128

data_loaders = {
    'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_data, batch_size=batch_size)
}
# 数据集大小
dataset_sizes = {
    'train': train_data.__len__(),
    'val': val_data.__len__()
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=16):
    # Todo: 这里应该是nn.Sigmoid还是nn.Sigmoid()?
    #  是nn.Sigmoid()
    sigmoid_func = nn.Sigmoid()
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            total_loss = 0.0
            total_precision = 0.0
            total_recall = 0.0
            batch_num = 0

            # 如果是训练阶段
            if phase == 'train':
                # 更新学习率
                # 进入训练模式
                model.train()
                total_batch = len(data_loaders[phase])
                # data表示一个批次的数据
                for data in data_loaders[phase]:
                    inputs, labels = data
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 梯度清零
                    optimizer.zero_grad()
                    # 前向传播
                    outputs = model(inputs)
                    # 这里返回的是平均损失
                    loss = criterion(sigmoid_func(outputs), labels)
                    precision, recall = calculate_accuracy_mode1(sigmoid_func(outputs), labels)
                    total_precision += precision
                    total_recall += recall
                    batch_num += 1
                    # 反向传播
                    loss.backward()
                    # 更新权重
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item() * inputs.size(0)

                    print('       progress: {}/{} '.format(batch_num, total_batch))
            # 验证阶段
            # Todo: 先不进行验证
            else:
                # 验证阶段不计算梯度
                with torch.no_grad():
                    # 进入预测模式
                    model.eval()
                    for data in data_loaders[phase]:
                        inputs, labels = data
                        if use_gpu:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                        outputs = model(inputs)
                        loss = criterion(sigmoid_func(outputs), labels)
                        total_loss += loss.item() * inputs.size(0)

                        precision, recall = calculate_accuracy_mode1(sigmoid_func(outputs), labels)
                        total_precision += precision
                        total_recall += recall
                        batch_num += 1
            epoch_loss = total_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            epoch_precision = total_precision / batch_num
            print('{} Precision: {:.4f} '.format(phase, epoch_precision))
            epoch_recall = total_recall / batch_num
            print('{} Recall: {:.4f} '.format(phase, epoch_recall))

            info = [epoch, epoch_loss, epoch_precision, epoch_recall]
            if phase == 'train':
                train_writer.writerow(info)
            else:
                val_writer.writerow(info)
            # Todo: 保存中间模型
            # torch.save(model.state_dict(), 'The_' + str(epoch) + '_epoch_model.pkl'"Themodel_AlexNet.pkl")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


# Todo: 定义准确率的方式，其实也是决策规则：按照精度阈值判决还是取topK
# 设定精度阈值，如果预测的概率值大于该阈值，则认为该图像含有这类标签
def calculate_accuracy_mode1(model_pred, labels):
    # 精度阈值
    accuracy_th = 0.5
    pred_res = model_pred > accuracy_th
    pred_res = pred_res.float()
    pred_sum = torch.sum(pred_res)
    if pred_sum == 0:
        return 0, 0
    target_sum = torch.sum(labels)
    # element-wise multiply
    true_predict_sum = torch.sum(pred_res * labels)
    # 模型预测的结果中有多少个结果正确
    precision = true_predict_sum / pred_sum
    # 模型预测正确的结果中，占样本真实标签的数量
    recall = true_predict_sum / target_sum
    return precision.item(), recall.item()


# 计算准确率的方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_accuracy_mode2(model_pred, labels):
    precision = 0
    recall = 0
    top = 3
    # Todo: debug此处，搞清楚数据维度和axis
    #  这是一个批次的数据：第一维是样本数量，第二维度是标签数量
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    # 为每一幅图像计算预测准确率和预测查全率
    for i in range(model_pred.shape[0]):
        tmp_label = torch.zeros(1, model_pred.shape[1])
        tmp_label[0, pred_label_locate[i]] = 1
        target_sum = torch.sum(labels[i])
        true_predict_sum = torch.sum(tmp_label * labels[i])
        precision += true_predict_sum / top
        recall += true_predict_sum / target_sum
    return precision, recall


# 对AlexNet模型进行修改
if __name__ == '__main__':
    model = models.alexnet(pretrained=True)
    # 获取最后一个全连接层的输入通道数，同时也是倒数第二层的输出通道数
    # 默认的最后一个层是4096*1000的全连接层，现在需要对输出进行修改：弹出层-->创建层
    num_input = model.classifier[6].in_features
    # 获取全连接层的网络结构
    feature_model = list(model.classifier.children())
    # 去掉原来的最后一层
    feature_model.pop()
    label_num = 90
    feature_model.append(nn.Linear(num_input, label_num))

    # 重新生成网络的后半部分
    model.classifier = nn.Sequential(*feature_model)

    if use_gpu:
        model = model.cuda()
    # 定义损失函数
    # Todo: 关于BCELoss的学习
    criterion = nn.BCELoss()

    # Todo: python中的id函数返回对象的唯一标识符
    fc_params = list(map(id, model.classifier[6].parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    params = [{"params": base_params, "lr": 0.0001},
              {"params": model.classifier[6].parameters(), "lr": 0.001}, ]
    optimizer_ft = torch.optim.SGD(params, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
