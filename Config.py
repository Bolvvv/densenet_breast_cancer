class TrainingConfig(object):
    batch_size = 8
    lr = 0.01
    epoches = 50
    show_n_iter = 10#执行多少个iter展示一次当前参数
    model_class = 169#densenet类型，121、161、169、201
    init_channel = 3#图片通道数
    classify = 3#分类数
    train = True
