import os
"""
对label文件进行筛选，去除不存在的数据
label文件格式：
数据名称 label
"""
def read_file(label_file, image_file):
    """
    读取源文件，并打印出不存在的数据，最后将存在的数据存储于[(),()...]中
    参数说明：
    label_file:形如：“../data/labels/train_data.txt”
    image_file:形如:“../data/img_file/”
    """
    image_label_list = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip().split()
            name = content[0]
            label = content[1]
            if os.path.exists(image_file+name):
                image_label_list.append((name, label))
            else:
                print("not find:"+name)
    return image_label_list

def write_file(file_list, file_name):
    """
    将read_file文件生成的数据写入到新的文件中
    参数说明：
    file_list:存储label文件的list，形如：[(str_img_name, str_label), (str_img_name, str_label)]
    file_name:形如：待写入label的文件名称，label.txt
    """
    with open(file_name, 'w') as f:
        for i in file_list:
            (name, label) = i
            f.writelines(name+" "+label+"\n")

l = read_file("../data/deepbc/labels_1218/test_BX.txt","../data/deepbc/usg_images_cutted_p1/")
write_file(l, "tmp_1.txt")
print(len(l))