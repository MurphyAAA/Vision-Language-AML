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
        if opt['train_clip'] == 'True': # fine-tune clip
            train_loader_source, train_loader_target, validation_loader, test_loader, train_clip_loader = build_splits_clip_disentangle(opt,experiment.preprocess)
            return experiment, train_loader_source, train_loader_target, validation_loader, test_loader, train_clip_loader
        else:
            train_loader_source, train_loader_target, validation_loader, test_loader = build_splits_clip_disentangle(opt,experiment.preprocess)
            return experiment, train_loader_source, train_loader_target, validation_loader, test_loader

    else:
        raise ValueError('Experiment not yet supported.')

def main(opt):
    fine_tune_clip_flag = False
    if opt['experiment'] == 'baseline':
        experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)
    elif opt['experiment'] == 'clip_disentangle' and opt['train_clip'] == 'True':
        experiment, train_loader_source, train_loader_target, validation_loader, test_loader, train_clip_loader = setup_experiment(opt)
        fine_tune_clip_flag = True
    else: # 'domain_disentangle' or (clip_disentangle && train_clip==false)
        experiment, train_loader_source, train_loader_target, validation_loader, test_loader = setup_experiment(opt)

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
        tot_l_clip =0

    # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):  # 如果有checkpoint 则加载
            epoch, iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logger1.info(opt)
        logger1.info('——————————————————————————————————————————————————————————————————') # logging.info() 输出到日志

        # Train loop
        while iteration < opt['max_iterations']:
            logger1.info(f'[epoch - {epoch}] ')
            if opt['experiment'] == 'baseline':
                for data in train_loader:
                    total_train_loss += experiment.train_iteration(data) # forward

                    if iteration % opt['print_every'] == 0: # print every 50 iteration
                        logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')

                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader)
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
                    if iteration % opt['print_every'] == 0:
                        # record loss info
                        logger1.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                        logger2.info(f'train_loss: {total_train_loss / (iteration + 1)}')
                        logger2.info(f'class_ent_loss: {tot_l_class_ent / (iteration + 1)}')
                        logger2.info(f'domain_ent_loss: {tot_l_domain_ent / (iteration + 1)}')
                        logger2.info(f'class_loss: {tot_l_class / (iteration + 1)}')
                        logger2.info(f'domain_loss: {tot_l_domain / (iteration + 1)}')
                        logger2.info(f'rec_loss: {tot_l_rec / (iteration + 1)}')
                        logger2.info('————————————————————————')
                    if iteration % opt['validate_every'] == 0: # validate every 100 iteration
                        # Run validation
                        val_accuracy, val_loss , mean_dom_accu = experiment.validate(validation_loader)
                        # record loss info
                        logger1.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        logger2.info(f'val_loss: {val_loss}')
                        logger2.info(f'dom_acc: {(100* mean_dom_accu):.2f}')
                        logger2.info(f'val_acc: {(100 * val_accuracy):.2f}')
                        logger2.info('———————————————————————————————————————————————')

                        if val_accuracy >= best_accuracy: # update best model
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

                if opt['train_clip'] =='True' and fine_tune_clip_flag == True: # fine tune CLIP.
                    fine_tune_clip_flag = False
                    os.makedirs(f'{opt["output_path"]}/clip_model', exist_ok=True)
                    clip_iteration = 0
                    clip_tot_loss=0
                    if os.path.exists(f'{opt["output_path"]}/clip_model/last_checkpoint.pth'):  # if already have best, load it
                        clip_iteration, clip_tot_loss = experiment.load_clip_checkpoint(f'{opt["output_path"]}/clip_model/last_checkpoint.pth')
                    print("fine-tune clip")
                    experiment.unfreeze_clip()
                    while clip_iteration < opt['clip_iteration']:
                        for data in train_clip_loader:
                            clip_tot_loss+=experiment.clip_train_iteration(data)
                            if clip_iteration % opt['print_every'] == 0:
                                print(f'[CLIP TRAIN - {clip_iteration}] Loss: {clip_tot_loss / (clip_iteration + 1)}')
                                logger2.info(f'clip_train_loss: {clip_tot_loss / (clip_iteration + 1)}')

                                experiment.save_clip_checkpoint(f'{opt["output_path"]}/clip_model/last_checkpoint.pth', clip_iteration, clip_tot_loss)
                            clip_iteration += 1

                            if clip_iteration > opt['clip_iteration']:
                                break
                    experiment.freeze_clip()
                    experiment.clip_model.float() # back to fp32 for later training
                    print("finish clip pre-training ",clip_iteration)
                len_dataloader = min(len(train_loader_source), len(train_loader_target))
                data_source_iter = iter(train_loader_source)
                data_target_iter = iter(train_loader_target)
                i = 0
                while i < len_dataloader:
                    data_source = next(data_source_iter)  # next(...)
                    data_target = next(data_target_iter)  # next(...)
                    tloss, l_class, l_class_ent, l_domain, l_domain_ent, L_rec, L_clip =experiment.train_iteration(data_source,data_target)  # forward
                    total_train_loss += tloss
                    tot_l_class_ent += l_class_ent
                    tot_l_domain_ent += l_domain_ent
                    tot_l_clip += L_clip
                    tot_l_class += l_class
                    tot_l_domain += l_domain
                    tot_l_rec += L_rec

                    if iteration % opt['print_every'] == 0:
                        logger1.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                        logger2.info(f'train_loss: {total_train_loss / (iteration + 1)}')
                        logger2.info(f'class_ent_loss: {tot_l_class_ent / (iteration + 1)}')
                        logger2.info(f'domain_ent_loss: {tot_l_domain_ent / (iteration + 1)}')
                        logger2.info(f'class_loss: {tot_l_class / (iteration + 1)}')
                        logger2.info(f'domain_loss: {tot_l_domain / (iteration + 1)}')
                        logger2.info(f'rec_loss: {tot_l_rec / (iteration + 1)}')
                        logger2.info(f'clip_loss: {tot_l_clip / (iteration + 1)}')
                        logger2.info('————————————————————————')
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss ,mean_dom_acc= experiment.validate(validation_loader)
                        logger1.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        logger2.info(f'val_loss: {val_loss}')
                        logger2.info(f'val_accuracy: {(100 * val_accuracy):.2f}')
                        logger2.info(f'mean_dom_acc: {(100 * mean_dom_acc):.2f}')
                        logger2.info('————————————————————————————————————————————————')
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

    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    if opt['experiment'] == 'domain_disentangle' or opt['experiment'] == "clip_disentangle":
        test_accuracy, _ , _  = experiment.validate(test_loader)
    else:
        test_accuracy, _   = experiment.validate(test_loader)
    # logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
    logger1.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f} (best model)')
    print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f} (best model)')

    experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
    if opt['experiment'] == 'domain_disentangle'or opt['experiment'] == "clip_disentangle":
        test_accuracy, _ ,_= experiment.validate(test_loader)
    else:
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
    log_file1 = opt["output_path"] + '/log.txt'
    log_file2 = opt["output_path"] + '/loss.txt'

    logger1 = get_logger('1', log_file1)
    logger2 = get_logger('2', log_file2)

    main(opt)
    print("---finish---")
