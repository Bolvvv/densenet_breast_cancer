import torch
from torch import utils
from CustomDataset import CustomDataset
from Config import TrainingConfig
from torchvision import datasets, transforms, models
import torch.nn as nn
import time

#设置cuda使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(model, config, train_loader, criterion, optimizer, epoch):
    model.train()

    # Train the model
    total_step = len(train_loader)
    correct = 0
    total = 0
    total_loss = 0#计算一个batch的所有loss
    n_iter_loss = 0#计算一个n_iter的所有loss

    for i ,(images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #计算模型准确率和损失
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item() #由于损失是均值，因此返回总的损失值

        n_iter_loss += loss.item()
        if (i+1) % config.show_n_iter == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.6f}'
                .format(epoch+1, config.epoches, i+1, total_step, n_iter_loss/(10 * labels.size(0))))
            n_iter_loss = 0#展示完成后清零
    sum_acc = 100 * correct / total
    sum_loss = total_loss / total#返回平均损失
    return sum_acc, sum_loss

        
        

def valid_step(model, config, valid_loader, criterion):
    model.eval()
    total_step = len(valid_loader)

    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0#计算一个batch的所有loss
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            #计算模型准确率和损失
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() #由于损失是均值，因此返回总的损失值
        sum_acc = 100 * correct / total
        sum_loss = total_loss / total
        print('Valid accuracy of the model on the valid images: %.4f '% (sum_acc))

    return sum_acc, sum_loss

def train_valid(model, config, train_loader, valid_loader):
    #记录最好准确率
    best_acc = 0

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    result_file_path = './result/'+time.strftime('%m%d_%H%M_%S',time.localtime(time.time()))+'_results.csv'
    #将信息写入csv文件zhong 
    with open(result_file_path, 'w') as f:
        f.write('batch_size %d, lr %f, epoches %d, model_class %s, start_time %s\n' % (config.batch_size, config.lr, config.epoches, config.model_class, time.strftime('%m%d_%H%M_%S',time.localtime(time.time()))))
        f.write('epoch,train_loss,train_acc,valid_loss,valid_acc\n')

    for epoch in range(config.epoches):
        train_acc, train_loss = train_step(model, config, train_loader, criterion, optimizer, epoch)
        valid_acc, valid_loss = valid_step(model, config, valid_loader, criterion)
        #进行模型评价，如果最好则存储模型
        if best_acc < valid_acc :
            best_acc = valid_acc
            print("New best Accuracy: %.4f" % best_acc)
            torch.save(model.state_dict(), './result/model.dat')
        #存储此次epoch训练结果
        with open(result_file_path, 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
            ))
    with open(result_file_path, 'a') as f:
        f.write('End Time%s\n' % (time.strftime('%m%d_%H%M_%S',time.localtime(time.time()))))


def test(test_loader):
    print("load model...")
    model.load_state_dict(torch.load('./result/model.dat'))#载入模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            #准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test accuracy of the model on the test images: %.4f '% (100 * correct / total))

def load_data(config):
    mean = [0.5,]
    stdv = [0.2,]
    train_transforms = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    train_set = CustomDataset(filename="../data/deepbc/labels_1218/train_CMGH.txt", image_dir="../data/deepbc/usg_images_cutted_v3", transform=train_transforms)
    valid_set = CustomDataset(filename="../data/deepbc/labels_1218/valid_CMGH.txt", image_dir="../data/deepbc/usg_images_cutted_v3", transform=test_transforms)
    test_set = CustomDataset(filename="../data/deepbc/labels_1218/valid_CMGH.txt", image_dir="../data/deepbc/usg_images_cutted_v3", transform=test_transforms)

    #载入训练参数
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()))

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    #获取训练超参
    config = TrainingConfig
    #载入数据
    train_loader, valid_loader, test_loader = load_data(config)

    #获取模型使用预训练参数
    if config.model_class == 121:
        model = models.densenet121(pretrained=True)
        model.features.conv0 = nn.Conv2d(config.init_channel, 64, kernel_size=7, stride=2,padding=3, bias=False)
    else:
        model.features.conv0 = nn.Conv2d(config.init_channel, 96, kernel_size=7, stride=2,padding=3, bias=False)
        if config.model_class == 161:
            model = models.densenet161(pretrained=True)
        elif config.model_class == 169:
            model = models.densenet169(pretrained=True)
        elif config.model_class == 201:
            model = models.densenet201(pretrained=True)
        else:
            print("模型选择错误！")
    
    #并修改原始通道数和分类数
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, config.classify)
    model.to(device)

    #模型训练与评价
    if config.train:
        train_valid(model, config, train_loader, valid_loader)
    else:
        test(test_loader)