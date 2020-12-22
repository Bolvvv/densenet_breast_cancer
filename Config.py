class TrainingConfig(object):
    batch_size = 36
    test_batch_size = 192
    lr = 0.001
    epoches = 200
    show_n_iter = 10#执行多少个iter展示一次当前参数
    model_class = 161#densenet类型，121、161、169、201
    init_channel = 3#图片通道数
    classify = 2#分类数
    train = False
