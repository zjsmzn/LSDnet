import os
import glob
import random
import json


train_dir = 'TRAIN DATA'
valid_dir = 'VAL DATA'
save_dir = 'SAVE PATH'
class_type1 = 'LYNCH'
class_type2 = 'NON_LYNCH'
num_count_class1 = 1 
num_count_class2 = 1


def write_txt(file_list, txt_path):
    with open(txt_path, 'w') as f:
        json.dump(file_list, f)


def make_list(file_dir, num_count_type):
    file_list = glob.glob(file_dir)
    all_list = []
    for file in file_list:
        file_num_list = os.listdir(file)
        temp=[]
        for num in file_num_list:
            temp.append(file+'/'+num)
        all_list.append(temp)
    return all_list


def run():
    mutation_train_dir = os.path.join(train_dir, class_type1)
    wild_train_dir = os.path.join(train_dir, class_type2)
    mutation_valid_dir = os.path.join(valid_dir, class_type1)
    wild_valid_dir = os.path.join(valid_dir, class_type2)

    train_list = make_list(mutation_train_dir, num_count_class1) + make_list(wild_train_dir, num_count_class2)
    valid_list = make_list(mutation_valid_dir, num_count_class1) + make_list(wild_valid_dir, num_count_class2)
    print('train length:', len(train_list), '   valid length:', len(valid_list))
    write_txt(train_list, os.path.join(save_dir, 'train_m{}_w{}.json'.format(str(num_count_class1), str(num_count_class2))))
    write_txt(valid_list, os.path.join(save_dir, 'valid_m{}_w{}.json'.format(str(num_count_class1), str(num_count_class2))))


if __name__ == '__main__':
    run()
