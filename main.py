import torch
from torch import utils
from CustomDataset import CustomDataset
from Config import TrainingConfig
from torchvision import datasets, transforms, models
import torch.nn as nn

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

    #将信息写入csv文件zhong 
    with open('results.csv', 'w') as f:
        f.write('epoch,train_loss,train_acc,valid_loss,valid_acc\n')

    for epoch in range(config.epoches):
        train_acc, train_loss = train_step(model, config, train_loader, criterion, optimizer, epoch)
        valid_acc, valid_loss = valid_step(model, config, valid_loader, criterion)
        #进行模型评价，如果最好则存储模型
        if best_acc < valid_acc :
            best_acc = valid_acc
            print("New best Accuracy: %.4f" % best_acc)
            torch.save(model.state_dict(), 'model.dat')
        #存储此次epoch训练结
        with open('results.csv', 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
            ))


def test(test_loader):
    model.load_state_dict(torch.load('model.dat'))#载入模型
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


if __name__ == "__main__":
    config = TrainingConfig
    mean = [0.5,]
    stdv = [0.2,]
    train_transforms = transforms.Compose([
        transforms.Resize((330, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((330, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    train_set = CustomDataset(filename="../data/inbreast/train_label.txt", image_dir="../data/inbreast/Inbreast", transform=train_transforms)
    valid_set = CustomDataset(filename="../data/inbreast/valid_label.txt", image_dir="../data/inbreast/Inbreast", transform=test_transforms)
    test_set = CustomDataset(filename="../data/inbreast/test_label.txt", image_dir="../data/inbreast/Inbreast", transform=test_transforms)

    #获取模型使用预训练参数并修改原始通道数和分类数
    model = models.densenet121(pretrained=True)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,padding=3, bias=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)

    #载入训练参数
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()))   

    #模型送入cuda
    model.to(device)
    #模型训练与评价
    train_valid(model, config, train_loader, valid_loader)