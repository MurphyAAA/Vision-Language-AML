import argparse
import torch

def parse_arguments():
    #  e.g.  python main.py --target_domain art_painting
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='baseline', choices=['baseline', 'domain_disentangle', 'clip_disentangle'])

    parser.add_argument('--target_domain', type=str, default='photo', choices=['art_painting', 'cartoon', 'sketch', 'photo'])
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--max_iterations', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--validate_every', type=int, default=100)

    if 0 == 0:
    # local version
        parser.add_argument('--output_path', type=str, default='.', help='Where to create the output directory containing logs and weights.')
        parser.add_argument('--data_path', type=str, default='data/PACS', help='Locate the PACS dataset on disk.')
    # collab version
    else:
         parser.add_argument('--output_path', type=str, default='./Vision-Language-AML/', help='Where to create the output directory containing logs and weights.')
         parser.add_argument('--data_path', type=str, default='./Vision-Language-AML/data/PACS', help='Locate the PACS dataset on disk.')

    parser.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU.')
    parser.add_argument('--test', action='store_true', help='If set, the experiment will skip training.')
    
    # Additional arguments can go below this line:
    #parser.add_argument('--test', type=str, default='some default value', help='some hint that describes the effect')
    parser.add_argument('--train_clip', type=str, default='False', choices=['True','False']) # True时自己手动预训练CLIP
    parser.add_argument('--clip_iteration', type=int, default=2000)
    # Build options dict
    opt = vars(parser.parse_args())

    if not opt['cpu']:
        assert torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'

    opt['output_path'] = f'{opt["output_path"]}/record/{opt["experiment"]}_{opt["target_domain"]}'

    return opt