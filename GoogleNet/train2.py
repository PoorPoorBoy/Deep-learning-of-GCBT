# --coding:utf-8--
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter
from model import GoogLeNet

import json

batch_size = 128
# 获得数据生成器，以字典的形式保存。
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(100),
        # transforms.CenterCrop(100),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(100),
        # transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = r'D:\wzh_data2\2class_new\dongmai_xianwei'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=0)  # 单线程
               for x in ['train', 'val']}

# 数据集的大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 类的名称
class_names = image_datasets['train'].classes
# 有GPU就用GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                     data_transforms['train'])
train_num = len(train_dataset)
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices2.json', 'w') as json_file:
    json_file.write(json_str)


# 模型训练和参数优化
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_iter = 0
    for epoch in range(num_epochs):
        epoch_iter += 1
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            train_iter = 0
            for inputs, labels in dataloaders[phase]:
                train_iter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model.aux_logits = False
                    outputs = model(inputs)
                    preds = torch.max(outputs, dim=1)[1]
                    model.aux_logits = True
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print('epoch:', epoch_iter, '当前已训练图片数量：', train_iter * batch_size, 'loss:', loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step(epoch_loss)
                optimizer.step()

            writer.add_scalar('loss_%s' % phase, epoch_loss, epoch)
            writer.add_scalar('acc_%s' % phase, epoch_acc, epoch)
            writer.add_histogram('loss', epoch_loss, epoch)
            writer.add_histogram('acc', epoch_acc, epoch)

            if epoch % 1 == 0:
                torch.save(model.state_dict(), "./models/googlenet_d_x_model-{}.pt".format(epoch))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model_ft = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True)
    """ 

    writer = SummaryWriter()
    '''冻结网络和参数
    for param in model_ft.parameters():
        param.requires_grad = False'''
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 4)  # 分类种类个数
    model_weight_path = "./models/model-80.pt"
    missing_keys, unexpected_keys = model_ft.load_state_dict(torch.load(model_weight_path), strict=False)


    model_weight_path = "./densenet121-a639ec97.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict载入模型权重。torch.load(model_weight_path)载入到内存当中还未载入到模型当中
    missing_keys, unexpected_keys = model_ft.load_state_dict(torch.load(model_weight_path), strict=False)

    '''冻结网络和参数
    for param in model_ft.parameters():
        param.requires_grad = False'''
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 3)  # 分类种类个数
    """

    writer = SummaryWriter()

    # 神经网络可视化
    images = torch.zeros(1, 3, 100, 100)  # 要求大小与输入图片的大小一致
    writer.add_graph(model_ft, images, verbose=False)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.005)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5, verbose=False,
                                                      threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                      eps=1e-08)  # 自适应学习率调整
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=1001)  # 训练次数

    writer.close()
    torch.save(model_ft.state_dict(), 'models/Densenet121_myself.pt')