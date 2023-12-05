import os
import random
from shutil import copyfile

def split_dataset_with_labels(input_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios should sum to 1.0"

    train_list = []
    val_list = []
    test_list = []

    for root, dirs, files in os.walk(input_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            output_train_dir = os.path.join(output_folder, 'train', dir_name)
            output_val_dir = os.path.join(output_folder, 'val', dir_name)
            output_test_dir = os.path.join(output_folder, 'test', dir_name)

            os.makedirs(output_train_dir, exist_ok=True)
            os.makedirs(output_val_dir, exist_ok=True)
            os.makedirs(output_test_dir, exist_ok=True)

            files_in_dir = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            random.shuffle(files_in_dir)

            num_files = len(files_in_dir)
            num_train = int(train_ratio * num_files)
            num_val = int(val_ratio * num_files)
            num_test = int(test_ratio * num_files)

            train_files = files_in_dir[:num_train]
            val_files = files_in_dir[num_train:num_train + num_val]
            test_files = files_in_dir[num_train + num_val:]

            for file_name in train_files:
                src = os.path.join(dir_path, file_name)
                dst = os.path.join(output_train_dir, file_name)
                copyfile(src, dst)
                train_list.append(f'{os.path.join("train", dir_name, file_name)} {dir_name}')

            for file_name in val_files:
                src = os.path.join(dir_path, file_name)
                dst = os.path.join(output_val_dir, file_name)
                copyfile(src, dst)
                val_list.append(f'{os.path.join("val", dir_name, file_name)} {dir_name}')

            for file_name in test_files:
                src = os.path.join(dir_path, file_name)
                dst = os.path.join(output_test_dir, file_name)
                copyfile(src, dst)
                test_list.append(f'{os.path.join("test", dir_name, file_name)} {dir_name}')

    # 生成列表文件
    with open(os.path.join(output_folder, 'train_list.txt'), 'w') as train_txt:
        train_txt.write('\n'.join(train_list))

    with open(os.path.join(output_folder, 'val_list.txt'), 'w') as val_txt:
        val_txt.write('\n'.join(val_list))

    with open(os.path.join(output_folder, 'test_list.txt'), 'w') as test_txt:
        test_txt.write('\n'.join(test_list))

# 指定输入文件夹和输出文件夹
input_folder = './Dataset'
output_folder = './SplitDataset'

# 执行划分函数
split_dataset_with_labels(input_folder, output_folder)
