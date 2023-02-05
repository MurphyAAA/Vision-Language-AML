from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import json

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}
DOMAINS = {
    'art_painting':0,
    'cartoon':1,
    'photo':2,
    'sketch':3,
}
DESCRIPTORS = {
    0: 'level of details',
    1: 'edges',
    2: 'color saturation',
    3: 'color shades',
    4: 'background',
    5: 'single instance',
    6: 'text',
    7: 'texture',
    8: 'perspective',
}
class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y  # tuple(图片的tensor，类别label)


class PACSDatasetDomainDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, yd = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, yd  # tuple(图片的tensor，类别label, domain的label)

class PACSDatasetDomainDisentangle_CLIP(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
            img_path, y, yd, descriptions = self.examples[index]
            x = self.transform(Image.open(img_path).convert('RGB'))
            return x, y, yd,descriptions   # tuple(图片的tensor，类别label, domain的label, descriptions)

class PACSDatasetDomainDisentangle_train_CLIP(Dataset):
    def __init__(self, examples, clip_preprocess):
        self.examples = examples
        self.clip_preprocess = clip_preprocess
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
            img_path, descriptions = self.examples[index]
            x = self.clip_preprocess(Image.open(img_path))

            return x,descriptions   # tuple(图片的tensor, descriptions)


def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f: # e.g. 打开./data/PACS/art_painting.txt文件
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3] # dog、elephant....
        category_idx = CATEGORIES[category_name] # 枚举： 某个类型字符串的名字 -> 数字编号
        image_name = line[4]
        # data/PACS/art_painting/dog/pic_001.jpg
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path] # example[i] 第i个类 所有图片数据的路径 [xxx/pic_0.jpg、xxx/pic_1.jpg ...]
        else:
            examples[category_idx].append(image_path)
    return examples

def read_lines_CLIP(data_path, domain_names): # data_path:./data/PACS
    examples = {}
    # value 是(path,description)的元组
    # 文件中是所有图片，但只要指定domain的图片，以数组形式返回，类别id是索引，内容是路径和des的元组
    txtFileNames=["groupe1AML", "groupe1DAAI", "groupe2AML", "groupe2DAAI", "groupe3AML", "groupe3DAAI", "groupe5AML", "groupe6AML", "second"]
    all_imgspath_des=[] # 所有文件中当前domain的 (图的地址, descriptions) tuple
    all_cate_id=[] # 每张图对应的category 数字形式
    for filename in txtFileNames:
        path_desc, categoryids = extract_images(f'{data_path}/text/{filename}.txt',domain_names,data_path)
        # print(path_desc) # 一个文件中的 imagePath-description pair
        # path_desc=[(path,desIndex+base_desc_index) for path, desIndex in path_desc]

        all_imgspath_des = list(all_imgspath_des) + list(path_desc) # 所有文件总的image-descriptions pair
        all_cate_id = list(all_cate_id) + list(categoryids) # 每个文件对应的category id

    for i in range(len(all_cate_id)): # 遍历path_desc
        if all_cate_id[i] not in examples.keys():
            examples[all_cate_id[i]] = [all_imgspath_des[i]] # example[i]:第i个类 所有图片数据的路径和description的元组 [(xxx/pic_0.jpg, description1), (xxx/pic_1.jpg, description2)... ]
        else:
            examples[all_cate_id[i]].append(all_imgspath_des[i])
    # examples[i] 表示第i类，其中元素为tuple: (image_path, description)
    # print("[2]",examples[1]) # 第1类
    return examples
def extract_images(file_path, domain_names,data_path):
    images_desc=[]
    categoryids=[]
    with open(file_path, 'r') as f:
        data = f.read()
        data = json.loads(data.replace("\'","\""))
    # desc_index =0
    for item in data:
        line = item['image_name'].strip().split()[0].split('/')
        domain = line[0]
        category = line[1]
        # if domain == domain_name:
        if domain in domain_names:
            # print(domain)
            img_path = f'{data_path}/kfold/'
            des=''
            for i,x in enumerate(item['descriptions']):  # concat n string parameters -> 1 string
                des = des + f'[{DESCRIPTORS[i]}]: {x}; '
            images_desc.append((img_path + item['image_name'], des))
            categoryids.append(CATEGORIES[category])
    # images_desc: 一个数组，每个元素是一个元组(image_path, description)
    # categoryids: 数组，每个元素代表该索引的元组的category label:[0,1,2,1...] 其中 0-dog, 1-elephant ...
    return images_desc, categoryids

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    # xxx_examples[i] 第i个类图片路径们
    source_examples = read_lines(opt['data_path'], source_domain) # opt['data_path']: "data/PACS"
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()} # 每个类别有多少张图， dict.items()返回字典的键值对
    source_total_examples = sum(source_category_ratios.values()) # source domain一共多少张图
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()} # source domain 中各个类别图片数据占总图数的比例 e.g. dog占18.5% elephant:12.45% ...

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation 验证集一共多少条数据

    train_examples = []
    val_examples = []
    test_examples = []
#examples_list 这个类别的图的路径
    for category_idx, examples_list in source_examples.items(): # key(类别id): val(图片路径)
        split_idx = round(source_category_ratios[category_idx] * val_split_length) # (N_k * N_vali) / N_total 第k类中分割出去为验证集的index
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():

        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_domain_disentangle(opt):  # x, y, yd
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    #————构建 examples数组
    # xxx_examples[i] 第i个类图片路径们
    source_examples = read_lines(opt['data_path'], source_domain)  # opt['data_path']: "data/PACS"
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}  # 每个类别有多少张图， dict.items()返回字典的键值对
    source_total_examples = sum(source_category_ratios.values())  # source domain一共多少张图
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}  # source domain 中各个类别图片数据占总图数的比例 e.g. dog占18.5% elephant:12.45% ...

    # target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}  # 每个类别有多少张图， dict.items()返回字典的键值对
    # target_total_examples = sum(target_category_ratios.values())  # source domain一共多少张图
    # target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}  # source domain 中各个类别图片数据占总图数的比例 e.g. dog占18.5% elephant:12.45% ...

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation 验证集一共多少条数据
    # val_split_length2 = target_total_examples * 0.2

    train_examples_source = [] # 从source domain来的，有category
    train_examples_target=[]   # 从target domain来的，没有category ，只有domain label，只能用来训练domain clf
    val_examples = [] # 用于验证category classifier效果，所以必须要有category label，只能是从source domain来
    test_examples = []

    for category_idx, examples_list in source_examples.items():  # key(类别id): val(图片路径)
        split_idx = round(source_category_ratios[category_idx] * val_split_length)  # (N_k * N_vali) / N_total 第k类中分割出去为验证集的index
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples_source.append([example, category_idx, 0])  # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx, 0])  # each pair is [path_to_img, class_label]

    for category_idx, examples_list in target_examples.items():
        # split_idx = round(target_category_ratios[category_idx] * val_split_length2)
        for i, example in enumerate(examples_list):
            # if i > split_idx:
            train_examples_target.append([example, category_idx, 1])  # each pair is [path_to_img, class_label]
            # else:
            #     val_examples.append([example, category_idx, DOMAINS[target_domain]])  # each pair is [path_to_img, class_label]

            test_examples.append([example, category_idx, 1])  # each pair is [path_to_img, class_label]

### ______

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    print("train_examples_source: ", len(train_examples_source))
    print("train_examples_target: ", len(train_examples_target))
    print("val_examples: ", len(val_examples))
    print("test_examples: ", len(test_examples))
    # Dataloaders
    train_loader_source = DataLoader(PACSDatasetDomainDisentangle(train_examples_source, train_transform), batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    train_loader_target = DataLoader(PACSDatasetDomainDisentangle(train_examples_target, train_transform), batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle(val_examples, eval_transform), batch_size=opt['batch_size']*2,
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size']*2,
                             num_workers=opt['num_workers'], shuffle=True)

    return train_loader_source, train_loader_target, val_loader, test_loader

def build_splits_clip_disentangle(opt,clip_preprocess):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    # ————构建 examples字典
    # xxx_examples[i] 表示第i类，其中元素为tuple: (image_path, description)
    # source_examples, source_labeled_descriptions = read_lines_CLIP(opt['data_path'], source_domain)  # opt['data_path']: "data/PACS"
    # target_examples, target_labeled_descriptions = read_lines_CLIP(opt['data_path'], target_domain)
    # source_examples = read_lines(opt['data_path'], source_domain)  # opt['data_path']: "data/PACS"
    # target_examples = read_lines(opt['data_path'], target_domain)
    clip_examples = read_lines_CLIP(opt['data_path'],['art_painting', 'cartoon', 'photo', 'sketch']) # 无所谓domain，有description的都用来train clip

    source_examples = read_lines_CLIP(opt['data_path'], [source_domain])  # opt['data_path']: "data/PACS"
    target_examples = read_lines_CLIP(opt['data_path'], [target_domain])
    # 带description的图片直接从read_lines就放入source example和target example，再——》放入train_examples和test_examples中，和普通训练集一样切开一部分放入验证集
    # print(source_examples)
    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in
                              source_examples.items()}  # 每个类别有多少张图， dict.items()返回字典的键值对
    source_total_examples = sum(source_category_ratios.values())  # source domain一共多少张图
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}  # source domain 中各个类别图片数据占总图数的比例 e.g. dog占18.5% elephant:12.45% ...

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation 验证集一共多少条数据

    train_clip_examples = []
    train_examples = []
    val_examples = []
    test_examples = []
    for category_idx, examples_list in clip_examples.items():  # key(类别id): val(图片路径, description id)
        for i, example in enumerate(examples_list): # example (图片路径, description)
            path, descriptions = example
            train_clip_examples.append([path, descriptions])  # each pair is [path_to_img, description]

    # images [without or with] descriptions
    for category_idx, examples_list in source_examples.items():  # key(类别id): val(图片路径, description id)
        split_idx = round(source_category_ratios[category_idx] * val_split_length)  # (N_k * N_vali) / N_total 第k类中分割出去为验证集的index
        for i, example in enumerate(examples_list): # example (图片路径, description)
            path, descriptions = example
            if i > split_idx:
                train_examples.append([path, category_idx, DOMAINS[source_domain], descriptions])  # each pair is [path_to_img, class_label, description]
            else:
                val_examples.append([path, category_idx, DOMAINS[source_domain], descriptions])  # each pair is [path_to_img, class_label, description]

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            path, descriptions = example
            test_examples.append([path, category_idx, DOMAINS[target_domain], descriptions])  # each pair is [path_to_img, class_label, description]

    ### ______

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    print("train_examples: ", len(train_examples))
    print("val_examples: ", len(val_examples))
    # Dataloaders
    if opt['train_clip'] == 'True':
        print("fine-tune clip examples: ", len(train_clip_examples))
        train_clip_loader = DataLoader(PACSDatasetDomainDisentangle_train_CLIP(train_clip_examples, clip_preprocess),batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)

    train_loader = DataLoader(PACSDatasetDomainDisentangle_CLIP(train_examples, train_transform),batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle_CLIP(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle_CLIP(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=False)

    if opt['train_clip'] == 'True':
        return train_loader, val_loader, test_loader, train_clip_loader
    else:
        return train_loader, val_loader, test_loader#, source_labeled_descriptions, target_labeled_descriptions
