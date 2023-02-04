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
        return experiment, train_loader, validation_loader, test_loader

    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        train_loader_source, train_loader_target, validation_loader, test_loader = build_splits_domain_disentangle(opt)
        return experiment, train_loader_source, train_loader_target, validation_loader, test_loader

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        if opt['train_clip'] == 'True':
            train_loader, validation_loader, test_loader, train_clip_loader = build_splits_clip_disentangle(opt,experiment.preprocess)
            return experiment, train_loader, validation_loader, test_loader, train_clip_loader
        else: # fine-tune clip
            train_loader, validation_loader, test_loader = build_splits_clip_disentangle(opt,experiment.preprocess)
            return experiment, train_loader, validation_loader, test_loader

    else:
        raise ValueError('Experiment not yet supported.')
    
    # return experiment, train_loader, validation_loader, test_loader


def main(opt):
    fine_tune_clip_flag = False
    if opt['experiment'] == 'domain_disentangle':
        experiment, train_loader_source, train_loader_target, validation_loader, test_loader = setup_experiment(opt)
    elif opt['experiment'] == 'clip_disentangle' and opt['train_clip'] == 'True':
        experiment, train_loader, validation_loader, test_loader, train_clip_loader = setup_experiment(opt)
        fine_tune_clip_flag = True
    else: # baseline or (clip_disentangle && train_clip==false)
        experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)
    # Skip training if '--test' flag is set
    if not opt['test']:
    # --test is not set
        iteration = 0
        epoch = 0
        best_accuracy = 0
        total_train_loss = 0
        tot_l_class_ent=0
        tot_l_domain_ent = 0
        tot_l_class = 0
        tot_l_domain = 0
        tot_l_rec = 0

    # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):  # 如果有checkpoint 则加载
            epoch, iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logger1.info(opt)
        logger1.info('——————————————————————————————————————————————————————————————————') # logging.info() 输出到日志

        # Train loop 运行N次也只能训练一次，而不是在上次最好的基础上继续训练
        while iteration < opt['max_iterations']: # 如果target domain特也放入训练接则一轮是125次(len(train_loader)=125) 一共5000/125=40 epoch     train_loader越小迭代的epoch数量越多
        # while epoch < opt['num_epochs']:
            # 扫一轮训练数据
            logger1.info(f'[epoch - {epoch}] ')
            if opt['experiment'] == 'baseline':
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
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', epoch, iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', epoch, iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break
            elif opt['experiment'] == 'domain_disentangle':
                len_dataloader = min(len(train_loader_source), len(train_loader_target))
                data_source_iter = iter(train_loader_source)
                data_target_iter = iter(train_loader_target)
                # data_target_iter = iter(test_loader)
                i = 0
                while i<len_dataloader:
                    data_source = next(data_source_iter)# next(...)
                    data_target = next(data_target_iter)# next(...)
                    tloss, l_class_ent, l_domain_ent,l_class, l_domain, l_rec = experiment.train_iteration(data_source, data_target)  # 前向反向传播，Adam优化模型  data 只从source domain中取出的
                    total_train_loss += tloss
                    tot_l_class_ent += l_class_ent
                    tot_l_domain_ent += l_domain_ent
                    tot_l_class += l_class
                    tot_l_domain += l_domain
                    tot_l_rec += l_rec
                    if iteration % opt['print_every'] == 0:  # 每50次 输出一条当前的平均损失
                        logger1.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                        logger2.info(f'train_loss: {total_train_loss / (iteration + 1)}')
                        logger2.info(f'class_ent_loss: {tot_l_class_ent / (iteration + 1)}')
                        logger2.info(f'domain_ent_loss: {tot_l_domain_ent / (iteration + 1)}')
                        logger2.info(f'class_loss: {tot_l_class / (iteration + 1)}')
                        logger2.info(f'domain_loss: {tot_l_domain / (iteration + 1)}')
                        logger2.info(f'rec_loss: {tot_l_rec / (iteration + 1)}')
                        print(tot_l_class_ent/ (iteration + 1), tot_l_domain_ent/ (iteration + 1))
                    if iteration % opt['validate_every'] == 0:
                        # Run validation 每100次训练 用验证集跑一次看看准确率
                        val_accuracy, val_loss , mean_dom_loss = experiment.validate(validation_loader)  # validate()中才有计算accuracy ，train只更新weight不计算accuracy
                        # test_accuracy, _ = experiment.validate(test_loader)  # validate()中才有计算accuracy ，train只更新weight不计算accuracy
                        # print(f'[TEST - {iteration}] | Accuracy: {(100 * test_accuracy):.2f}')
                        logger1.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        logger2.info(f'val_loss: {val_loss}')
                        logger2.info(f'dom_acc: {mean_dom_loss}')
                        logger2.info(f'val_acc: {val_accuracy}')
                        if val_accuracy >= best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', epoch, iteration,
                                                       best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', epoch, iteration,
                                                   best_accuracy, total_train_loss)

                    iteration += 1
                    i += 1
                    if iteration > opt['max_iterations']:
                        break
            elif opt['experiment'] == 'clip_disentangle':
                # 先 手动预训练CLIP 用所有domain的作为数据集，而不仅仅是source domain了
                if opt['train_clip'] =='True' and fine_tune_clip_flag == True: # 需要手动训练clip，且还没训练过
                    fine_tune_clip_flag = False
                    clip_iteration = 0
                    print("fine-tune clip")
                    experiment.unfreeze_clip()
                    clip_tot_loss=0
                    while clip_iteration < opt['clip_iteration']:
                        for data in train_clip_loader:
                            clip_tot_loss+=experiment.clip_train_iteration(data)
                            if clip_iteration % opt['print_every'] == 0:
                                print(f'[CLIP TRAIN - {clip_iteration}] Loss: {clip_tot_loss / (clip_iteration + 1)}')
                            clip_iteration += 1

                            if clip_iteration > opt['clip_iteration']:
                                break
                    experiment.freeze_clip()
                    experiment.clip_model.float()
                    print("finish clip pre-training ",clip_iteration)
                len_dataloader = min(len(train_loader), len(test_loader)) # 数据少 扫一遍数据跑的iteration少
                data_source_iter = iter(train_loader)
                data_target_iter = iter(test_loader)
                i = 0
                while i < len_dataloader:
                    data_source = next(data_source_iter)  # next(...)
                    data_target = next(data_target_iter)  # next(...)
                    total_train_loss += experiment.train_iteration(data_source,data_target)  # 前向反向传播，Adam优化模型  data 只从source domain中取出的

                    if iteration % opt['print_every'] == 0:  # 每50次 输出一条当前的平均损失
                        logger1.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                        logger2.info(f'train1_loss: {total_train_loss / (iteration + 1)}')
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader)  # validate()中才有计算accuracy ，train只更新weight不计算accuracy
                        # print(len(validation_loader))
                        # print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        logger1.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        logger2.info(f'val_loss: {val_loss}')
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', epoch, iteration,
                                                       best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', epoch, iteration,
                                                   best_accuracy, total_train_loss)

                    iteration += 1
                    i += 1
                    if iteration > opt['max_iterations']:
                        break
            epoch += 1
            # if epoch >= opt['num_epochs']:
            #     break

    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    # logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
    logger1.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f} (best model)')
    print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f} (best model)')

    experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logger1.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f} (last model)')
    print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f} (last model)')

def get_logger(logger_name, file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)



if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)
    # print(opt["output_path"]) #./record/baseline_cartoon
    # Setup logger 绑定日志文件到 output_path/log.txt      level: debug(调试信息)/info(正常运行的信息)/warning(未来可能出的错)/error(某些功能不能继续)/critical(程序崩了)
   #logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    log_file1 = opt["output_path"] + '/log.txt'
    log_file2 = opt["output_path"] + '/loss.txt'

    logger1 = get_logger('1', log_file1)
    logger2 = get_logger('2', log_file2)

    main(opt)
    print("---finish---")
