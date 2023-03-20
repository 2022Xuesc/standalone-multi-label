from dataset_generator import *

src_dir = '../dataset/ms-coco/src_val'
guest_train_dir = '../dataset/ms-coco/guest/train'
host_train_dir = '../dataset/ms-coco/host/train'

guest_val_dir = '../dataset/ms-coco/guest/val'
host_val_dir = '../dataset/ms-coco/host/val'
dirs = [guest_train_dir, guest_val_dir, host_train_dir, host_val_dir]


# 共40000张图片，每个客户端训练2000张，测试200张


# generate_2014(src_dir, 0, 0.05, guest_train_dir)
# generate_2014(src_dir, 0.05, 0.1, host_train_dir)
# #
# generate_2014(src_dir, 0.1, 0.105, guest_val_dir)
# generate_2014(src_dir, 0.105, 0.11, host_val_dir)
#
#
# generate_labels(dirs)
#
# generate_configs(dirs)

# 绘制客户端训练数据集的直方图
def draw_histogram(data_dir):
    labels_path = os.path.join(data_dir, 'labels.txt')
    fp = open(labels_path, 'r')
    cnts = [0] * 90
    for line in fp:
        line.strip('\n')
        info = line.split(',')
        for index in range(1, len(info)):
            if info[index] == '1':
                cnts[index - 1] += 1
    print(cnts)


draw_histogram(guest_train_dir)