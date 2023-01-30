from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation 验证集一共多少条数据

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():  # key(类别id): val(图片路径)
        split_idx = round(source_category_ratios[category_idx] * val_split_length)  # (N_k * N_vali) / N_total 第k类中分割出去为验证集的index
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx, DOMAINS[source_domain]])  # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx, DOMAINS[source_domain]])  # each pair is [path_to_img, class_label]

    for category_idx, examples_list in target_examples.items():
        # split_idx = round(source_category_ratios[category_idx] * val_split_length) # (N_k * N_vali) / N_total 第k类中分割出去为验证集的index

        for example in examples_list:
            # if i>split_idx:
            # train_examples.append([example, -1, DOMAINS[target_domain]])
            # else:
            #     val_examples.append([example, -1, DOMAINS[target_domain]])
            test_examples.append([example, category_idx, DOMAINS[target_domain]])  # each pair is [path_to_img, class_label]
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
    train_loader = DataLoader(PACSDatasetDomainDisentangle(train_examples, train_transform), batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_clip_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    # ————构建 examples数组
    # xxx_examples[i] 第i个类图片路径们
    source_examples = read_lines(opt['data_path'], source_domain)  # opt['data_path']: "data/PACS"
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in
                              source_examples.items()}  # 每个类别有多少张图， dict.items()返回字典的键值对
    source_total_examples = sum(source_category_ratios.values())  # source domain一共多少张图
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}  # source domain 中各个类别图片数据占总图数的比例 e.g. dog占18.5% elephant:12.45% ...

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation 验证集一共多少条数据

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():  # key(类别id): val(图片路径)
        split_idx = round(
            source_category_ratios[category_idx] * val_split_length)  # (N_k * N_vali) / N_total 第k类中分割出去为验证集的index
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append(
                    [example, category_idx, DOMAINS[source_domain]])  # each pair is [path_to_img, class_label]
            else:
                val_examples.append(
                    [example, category_idx, DOMAINS[source_domain]])  # each pair is [path_to_img, class_label]

    for category_idx, examples_list in target_examples.items():
        # split_idx = round(source_category_ratios[category_idx] * val_split_length) # (N_k * N_vali) / N_total 第k类中分割出去为验证集的index

        for example in examples_list:
            # if i>split_idx:
            # train_examples.append([example, -1, DOMAINS[target_domain]])
            # else:
            #     val_examples.append([example, -1, DOMAINS[target_domain]])
            test_examples.append(
                [example, category_idx, DOMAINS[target_domain]])  # each pair is [path_to_img, class_label]
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
    train_loader = DataLoader(PACSDatasetDomainDisentangle(train_examples, train_transform),
                              batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader
