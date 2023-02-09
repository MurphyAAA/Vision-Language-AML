import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import json

from torchvision import transforms

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
        return x, y  # tuple(image tensor, category label)


class PACSDatasetDomainDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, yd = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, yd  # tuple(image, label, domain-label)



class PACSDatasetDomainDisentangle_CLIP(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
            img_path, y, yd, descriptions = self.examples[index]
            img = Image.open(img_path).convert('RGB')
            x = self.transform(img)
            return x, y, yd,descriptions   # tuple(image, label, domain-label, descriptions)

class PACSDatasetDomainDisentangle_train_CLIP(Dataset):
    def __init__(self, examples, clip_preprocess):
        self.examples = examples
        self.clip_preprocess = clip_preprocess
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
            img_path, descriptions = self.examples[index]
            x = self.clip_preprocess(Image.open(img_path))

            return x,descriptions   # tuple(image, descriptions)


def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f: # e.g. ./data/PACS/art_painting.txt
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3] # dog、elephant....
        category_idx = CATEGORIES[category_name] # text name -> index
        image_name = line[4]

        # data/PACS/art_painting/dog/pic_001.jpg
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path] # example[i] i-th category: [xxx/pic_0.jpg、xxx/pic_1.jpg ...]
        else:
            examples[category_idx].append(image_path)
    return examples

def read_lines_CLIP(data_path, domain_names): # data_path:./data/PACS
    examples = {} #key: category id, value: list of (image path, description)
    txtFileNames=["groupe1AML", "groupe1DAAI", "groupe2AML", "groupe2DAAI", "groupe3AML", "groupe3DAAI", "groupe5AML", "groupe6AML", "second", "photo", "sketch"]
    all_imgspath_des=[] # (image path, description) pair of current domain of all files
    all_cate_id=[] # category id of each image
    for filename in txtFileNames:
        path_desc, categoryids = extract_images(f'{data_path}/text/{filename}.txt',domain_names,data_path)
        all_imgspath_des = list(all_imgspath_des) + list(path_desc)
        all_cate_id = list(all_cate_id) + list(categoryids)

    for i in range(len(all_cate_id)): # traverse path_desc
        if all_cate_id[i] not in examples.keys():
            examples[all_cate_id[i]] = [all_imgspath_des[i]] # example[i]: i-th category: [(xxx/pic_0.jpg, description1), (xxx/pic_1.jpg, description2)... ]
        else:
            examples[all_cate_id[i]].append(all_imgspath_des[i])
    return examples
def extract_images(file_path, domain_names,data_path):
    images_desc=[]
    categoryids=[]
    with open(file_path, 'r') as f:
        data = f.read()
        data = json.loads(data.replace("\'","\""))
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
                des = des + f'{DESCRIPTORS[i]}: {x}; '
            images_desc.append((img_path + item['image_name'], des))
            categoryids.append(CATEGORIES[category])
    # images_desc: a list, elements are tuples: (image_path, description)
    # categoryids: a list, elements are category label i.e. [0,1,2,1...]. (0-dog, 1-elephant ...
    return images_desc, categoryids

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    # xxx_examples[i] images path of i-th category
    source_examples = read_lines(opt['data_path'], source_domain) # opt['data_path']: "data/PACS"
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation 验证集一共多少条数据

    train_examples = []
    val_examples = []
    test_examples = []
    for category_idx, examples_list in source_examples.items(): # key(category id): val(image path)
        split_idx = round(source_category_ratios[category_idx] * val_split_length) # (N_k * N_vali) / N_total
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
    #————build examples list
    source_examples = read_lines(opt['data_path'], source_domain)  # opt['data_path']: "data/PACS"
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation

    train_examples_source = []
    train_examples_target=[]   # for training domain clf
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)  # (N_k * N_vali) / N_total
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples_source.append([example, category_idx, 0])  # each pair is [path_to_img, class_label, domain(source)]
            else:
                val_examples.append([example, category_idx, 0])  # each pair is [path_to_img, class_label, domain(source)]

    for category_idx, examples_list in target_examples.items():
        for i, example in enumerate(examples_list):
            train_examples_target.append([example, category_idx, 1])  # each pair is [path_to_img, class_label, domain(target)]
            test_examples.append([example, category_idx, 1])  # each pair is [path_to_img, class_label, domain(target)]

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
    val_loader = DataLoader(PACSDatasetDomainDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=True)

    return train_loader_source, train_loader_target, val_loader, test_loader

def build_splits_clip_disentangle(opt,clip_preprocess):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    source_examples = read_lines(opt['data_path'], source_domain)  # opt['data_path']: "data/PACS"
    target_examples = read_lines(opt['data_path'], target_domain)
    # clip_examples = read_lines_CLIP(opt['data_path'],['art_painting', 'cartoon', 'photo', 'sketch']) #一会试一下只留source 和target
    clip_examples = read_lines_CLIP(opt['data_path'],['art_painting', target_domain])

    source_examples_des = read_lines_CLIP(opt['data_path'], [source_domain])  # opt['data_path']: "data/PACS"
    target_examples_des = read_lines_CLIP(opt['data_path'], [target_domain])
    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation

    train_clip_examples = []
    train_examples_source = []
    train_examples_target = []
    val_examples = []
    test_examples = []
    # for fine-tune CLIP
    for category_idx, examples_list in clip_examples.items():  # key(category id): val(image path, description id)
        for i, example in enumerate(examples_list): # example (image path, description)
            path, descriptions = example
            train_clip_examples.append([path, descriptions])  # each pair is [path_to_img, description]
    #-------------------
    # images [without] descriptions
    for category_idx, examples_list in source_examples.items():  # key(category id): val(image path, description id)
        split_idx = round(source_category_ratios[category_idx] * val_split_length)  # (N_k * N_vali) / N_total
        for i, example in enumerate(examples_list): # example (image path, description)
            if i > split_idx:
                train_examples_source.append([example, category_idx, 0, '-1'])  # each pair is [path_to_img, class_label, domain, description]
            else:
                val_examples.append([example, category_idx, 0, '-1'])  # each pair is [path_to_img, class_label, domain, description]

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            train_examples_target.append([example, category_idx, 1, '-1'])
            test_examples.append([example, category_idx, 1, '-1'])

    # images [with] descriptions
    for category_idx, examples_list in source_examples_des.items():
       for i, example in enumerate(examples_list):
            path, descriptions = example
            train_examples_source.append([path, category_idx, 0, descriptions])

    for category_idx, examples_list in target_examples_des.items():
        for example in examples_list:
            path, descriptions = example
            train_examples_target.append([path, category_idx, 1, descriptions])
            test_examples.append([path, category_idx, 1, descriptions])

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
    print("train_examples: ", len(train_examples_source))
    print("train_examples: ", len(train_examples_target))
    print("val_examples: ", len(val_examples))
    # Dataloaders
    if opt['train_clip'] == 'True':
        print("fine-tune clip examples: ", len(train_clip_examples))
        train_clip_loader = DataLoader(PACSDatasetDomainDisentangle_train_CLIP(train_clip_examples, clip_preprocess),batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)

    train_loader_source = DataLoader(PACSDatasetDomainDisentangle_CLIP(train_examples_source, train_transform),batch_size=opt['batch_size']//2,
                              num_workers=opt['num_workers'], shuffle=True)
    train_loader_target = DataLoader(PACSDatasetDomainDisentangle_CLIP(train_examples_target, train_transform), batch_size=opt['batch_size']//2,
                                     num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle_CLIP(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle_CLIP(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=True)

    if opt['train_clip'] == 'True':
        return train_loader_source, train_loader_target, val_loader, test_loader, train_clip_loader
    else:
        return train_loader_source, train_loader_target, val_loader, test_loader
