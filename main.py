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

    #显示参数设定
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
        total_loss += loss.item()

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
    #将信息写入csv文件中
    with open(result_file_path, 'w') as f:
        f.write('batch_size %d, lr %f, epoches %d, model_class %s, start_time %s\n' % (config.batch_size, config.lr, config.epoches, config.model_class, time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))))
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
        f.write('End Time %s\n' % (time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))))

def test(model, test_loader):
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

def save_full_model(model, model_src_path, model_save_path):
    """
    将参数存储为完整模型
    model:未实例化的模型
    model_src_path:模型参数
    model_save_path:完整模型存储地址
    """
    print("load model...")
    model.load_state_dict(torch.load(model_src_path))#载入参数
    torch.save(model, model_save_path)

def load_data(config):
    mean = [0.5,]
    stdv = [0.2,]
    train_transforms = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    #v3的测试数据集
    # label_path = "../data/deepbc/labels_1218/"
    # image_path = "../data/deepbc/usg_images_cutted_v3"
    # train_set_list = [label_path+"train_BX.txt", label_path+"train_CMGH.txt", label_path+"train_Malignant_DeYang.txt", label_path+"train_WestChina.txt"]
    # valid_set_list = [label_path+"valid_BX.txt", label_path+"valid_CMGH.txt", label_path+"valid_Malignant_DeYang.txt", label_path+"valid_WestChina.txt"]
    # test_set_list = [label_path+"test_BX.txt", label_path+"test_CMGH.txt", label_path+"test_Malignant_DeYang.txt", label_path+"test_WestChina.txt"]

    #p1的测试数据集
    label_path = "../data/deepbc/labels_photo/"
    image_path = "../data/deepbc/usg_images_cutted_p1"
    train_set_list = [label_path+"train_WestChina.txt"]
    valid_set_list = [label_path+"valid_WestChina.txt"]
    test_set_list = [label_path+"test_WestChina.txt"]

    train_set = CustomDataset(data_set_list=train_set_list, image_dir=image_path, transform=train_transforms)
    valid_set = CustomDataset(data_set_list=valid_set_list, image_dir=image_path, transform=test_transforms)
    test_set = CustomDataset(data_set_list=test_set_list, image_dir=image_path, transform=test_transforms)

    #载入训练参数
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.test_batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()))

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    #获取训练超参
    config = TrainingConfig
    #载入数据
    train_loader, valid_loader, test_loader = load_data(config)

    #模型参数配置
    if config.model_class == 161:
        model = models.densenet161(pretrained=True)
        model.features.conv0 = nn.Conv2d(config.init_channel, 96, kernel_size=7, stride=2,padding=3, bias=False)#修改原始通道数
    else:
        if config.model_class == 121:
            model = models.densenet121(pretrained=True)
        elif config.model_class == 169:
            model = models.densenet169(pretrained=True)
        elif config.model_class == 201:
            model = models.densenet201(pretrained=True)
        else:
            print("模型选择错误！")
        model.features.conv0 = nn.Conv2d(config.init_channel, 64, kernel_size=7, stride=2,padding=3, bias=False)#修改原始通道数
    #修改分类数
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, config.classify)
    model.to(device)

    #模型训练与评价
    if config.train:
        train_valid(model, config, train_loader, valid_loader)
    else:
        test(model, test_loader)
    #存储完整模型
    # save_full_model(model, './result/v3_result/model_161_93.67.dat', './best_model.pkl')