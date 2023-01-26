import os
import logging
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment

def setup_experiment(opt):
    
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt) # 实例化一个BaselineExperiment类 对象
        train_loader, validation_loader, test_loader = build_splits_baseline(opt) # DataLoader() 构建若干batch数据
        
    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_domain_disentangle(opt)

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_clip_disentangle(opt)

    else:
        raise ValueError('Experiment not yet supported.')
    
    return experiment, train_loader, validation_loader, test_loader


def main(opt):
    experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)

    # Skip training if '--test' flag is set
    if not opt['test']:
    # --test is not set
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):  # 如果有checkpoint 则加载
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)
        logging.info('——————————————————————————————————————————————————————————————————') # logging.info() 输出到日志

        # Train loop 运行N次也只能训练一次，而不是在上次最好的基础上继续训练
        while iteration < opt['max_iterations']: # 如果target domain特也放入训练接则一轮是125次(len(train_loader)=125) 一共5000/125=40 epoch     train_loader越小迭代的epoch数量越多
            # 扫一轮训练数据
            for data in train_loader: # Domain Distanglement的 train_loader必须包含domain的
                total_train_loss += experiment.train_iteration(data) # 前向反向传播，Adam优化模型  data 只从source domain中取出的

                if iteration % opt['print_every'] == 0: # 每50次 输出一条当前的平均损失
                    logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')

                if iteration % opt['validate_every'] == 0:
                    # Run validation
                    val_accuracy, val_loss = experiment.validate(validation_loader) # validate()中才有计算accuracy ，train只更新weight不计算accuracy
                    # print(len(validation_loader))
                    logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                    experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                iteration += 1
                if iteration > opt['max_iterations']:
                    break

    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)
    # print(opt["output_path"]) #./record/baseline_cartoon
    # Setup logger 绑定日志文件到 output_path/log.txt      level: debug(调试信息)/info(正常运行的信息)/warning(未来可能出的错)/error(某些功能不能继续)/critical(程序崩了)
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)
    print("---finish---")
