#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:16:29 2019

@author: abdel62
"""
import os
import sys
import importlib
import time
import datetime
import numpy as np

import torch
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.args_parser import args_parser, save_args, print_args, initialize_args,  compare_args_w_json
from dataloaders.dataloader_creator import create_dataloader
from utils.error_metrics import AverageMeter, create_error_metric, LogFile
from utils.save_output_images import create_out_image_saver, colored_depthmap_tensor
from utils.checkpoints import save_checkpoint
from common.losses import get_loss_fn
from utils.eval_uncertainty import eval_ause
import matplotlib.image as Image

def main():
    # Make some variable global
    global args, train_csv, test_csv, exp_dir, best_result, device, tb_writer, tb_freq

    # Args parser 
    args = args_parser()

    start_epoch = 0
############ EVALUATE MODE ############
    if args.evaluate:  # Evaluate mode
        print('\n==> Evaluation mode!')

        # Define paths
        chkpt_path = args.evaluate

        # Check that the checkpoint file exist
        assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)

        # Experiment director
        exp_dir = os.path.dirname(os.path.abspath(chkpt_path))
        sys.path.append(exp_dir)

        # Load checkpoint
        print('- Loading checkpoint:', chkpt_path)

        # Load the checkpoint
        checkpoint = torch.load(chkpt_path)

        # Assign some local variables
        args = checkpoint['args']
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        print('- Checkpoint was loaded successfully.')

        # Compare the checkpoint args with the json file in case I wanted to change some args
        compare_args_w_json(args, exp_dir, start_epoch+1)
        args.evaluate = chkpt_path

        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        model = checkpoint['model'].to(device)

        print_args(args)

        train_loader, val_loader = create_dataloader(args, eval_mode=True)

        loss = get_loss_fn(args).to(device)
        evaluate_epoch(train_loader, model, loss, start_epoch,train=True)
        evaluate_epoch(val_loader, model, loss, start_epoch)

        return  # End program

############ RESUME MODE ############
    elif args.resume:  # Resume mode
        print('\n==> Resume mode!')

        # Define paths
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)

        # Experiment directory
        exp_dir = os.path.dirname(os.path.abspath(chkpt_path))
        sys.path.append(exp_dir)

        # Load checkpoint
        print('- Loading checkpoint:', chkpt_path)
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        print('- Checkpoint ({}) was loaded successfully!\n'.format(checkpoint['epoch']))

        # Compare the checkpoint args with the json file in case I wanted to change some args
        compare_args_w_json(args, exp_dir, start_epoch)
        args.resume = chkpt_path

        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        model = checkpoint['model'].to(device)
        optimizer = checkpoint['optimizer']

        print_args(args)

        train_loader, val_loader = create_dataloader(args, eval_mode=False)

############ NEW EXP MODE ############
    else:  # New Exp
        print('\n==> Starting a new experiment "{}" \n'.format(args.exp))

        # Check if experiment exists
        ws_path = os.path.join('workspace/', args.workspace)
        exp = args.exp
        exp_dir = os.path.join(ws_path, exp)
        assert os.path.isdir(exp_dir), '- Experiment "{}" not found!'.format(exp)

        # Which device to use
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

        # Add the experiment's folder to python path
        sys.path.append(exp_dir)

        print_args(args)

        # Create dataloader
        train_loader, val_loader = create_dataloader(args, eval_mode=False)

        # import the model
        f = importlib.import_module('network')
        model = f.CNN().to(device)
        print('\n==> Model "{}" was loaded successfully!'.format(model.__name__))

        # Optimize only parameters that requires_grad
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        # Create Optimizer
        if args.optimizer.lower() == 'sgd':
            optimizer = SGD(parameters, lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'adam':
            optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

############ IF RESUME/NEW EXP ############
    # Error metrics that are set to the worst
    best_result = create_error_metric(args)
    best_result.set_to_worst()

    # Tensorboard
    tb = args.tb_log if hasattr(args, 'tb_log') else False
    tb_freq = args.tb_freq if hasattr(args, 'tb_freq') else 1000
    tb_writer = None
    if tb:
        tb_writer = SummaryWriter(os.path.join(exp_dir, 'tb_log', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    # Create Loss
    loss = get_loss_fn(args).to(device)

    # Define Learning rate decay
    lr_decayer = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor, last_epoch=start_epoch-1)

    # Create or Open Logging files
    train_csv = LogFile(os.path.join(exp_dir, 'train.csv'), args)
    test_csv = LogFile(os.path.join(exp_dir, 'test.csv'), args)
    best_txt = os.path.join(exp_dir, 'best.txt')

    save_args(exp_dir, args)  # Save args to JSON file

############ TRAINING LOOP ############
    for epoch in range(start_epoch, args.epochs):
            print('\n==> Training Epoch [{}] (lr={})'.format(epoch, optimizer.param_groups[0]['lr']))

            train_err_avg = train_epoch(train_loader, model, optimizer, loss, epoch)

            # Learning rate scheduler
            lr_decayer.step()

            train_csv.update_log(train_err_avg, epoch)

            # Save checkpoint in case evaluation crashed
            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'model': model,
                'best_result': best_result,
                'optimizer': optimizer,
            }, False, epoch, exp_dir)

            # Evaluate the trained epoch
            test_err_avg, out_image = evaluate_epoch(val_loader, model, loss, epoch)  # evaluate on validation set

            # Evaluate Uncerainty
            ause = None
            # if args.eval_uncert:
            #     if args.loss == 'masked_prob_loss_var':
            #         ause, ause_fig = eval_ause(model, val_loader, args, epoch, uncert_type='v')
            #     else:
            #         ause, ause_fig = eval_ause(model, val_loader, args, epoch, uncert_type='c')

            # Log to tensorboard if enabled
            if tb_writer is not None:
                avg_meter = test_err_avg.get_avg()
                tb_writer.add_scalar('Loss/val', avg_meter.loss, epoch)
                tb_writer.add_scalar('MAE/val', avg_meter.metrics['mae'], epoch)
                tb_writer.add_scalar('RMSE/val', avg_meter.metrics['rmse'], epoch)
                if ause is not None:
                    tb_writer.add_scalar('AUSE/val', ause, epoch)
                # tb_writer.add_images('Prediction', colored_depthmap_tensor(out_image[:, :1, :, :]), epoch)
                # tb_writer.add_images('Input_Conf_Log_Scale', colored_depthmap_tensor(torch.log(out_image[:, 2:, :, :]+1)), epoch)
                # tb_writer.add_images('Output_Conf_Log_Scale', colored_depthmap_tensor(torch.log(out_image[:, 5:6, :, :]+1)), epoch)
                # tb_writer.add_figure('Sparsification_Plot', ause_fig, epoch)

            # Update Log files
            test_csv.update_log(test_err_avg, epoch, ause)

            # Save best model
            # TODO: How to decide the best based on dataset?
            is_best = test_err_avg.metrics['rmse'] < best_result.metrics['rmse']
            if is_best:
                best_result = test_err_avg  # Save the new best locally
                test_err_avg.print_to_txt(best_txt, epoch)  # Print to a text file

            # Save it again if it is best checkpoint
            save_checkpoint({
                    'args': args,
                    'epoch': epoch,
                    'model': model,
                    'best_result': best_result,
                    'optimizer': optimizer,
                }, is_best, epoch, exp_dir)
            # TODO: Do you really need to save the best out_image ??


############ TRAINING FUNCTION ############
def train_epoch(dataloader, model, optimizer, objective, epoch):
    """
    Training function 
    
    Args:
        dataloader: The dataloader object for the dataset
        model: The model to be trained
        optimizer: The optimizer to be used
        objective: The objective function
        epoch: What epoch to start from
    
    Returns:
        AverageMeter() object.
    
    Raises:
        KeyError: Raises an exception.
    """
    err = create_error_metric(args)
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model.train()  # switch to train mode

    start = time.time()

    # visualizations of range image
    # for i, (input, target) in enumerate(dataloader):
    #         # plot
    #         range_image_range = input['proj'][0,0].numpy()
    #         range_image_x = input['proj'][0,1].numpy()
    #         range_image_y = input['proj'][0,2].numpy()
    #         range_image_z = input['proj'][0,3].numpy()
    #         range_image_remission = input['proj'][0,4].numpy()
    #         range_image_range = (range_image_range / range_image_range.max() * 255.).astype('int')
    #         Image.imsave('imgs/'+str(i)+'_'+"kitti_lidar.png", range_image_range)
    #
    #         range_image_range = target['proj'][0,0].numpy()
    #         range_image_x = target['proj'][0,1].numpy()
    #         range_image_y = target['proj'][0,2].numpy()
    #         range_image_z = target['proj'][0,3].numpy()
    #         range_image_remission = target['proj'][0,4].numpy()
    #         range_image_range = (range_image_range / range_image_range.max() * 255.).astype('int')
    #         Image.imsave('imgs/'+str(i)+'_'+"high_res_lidar.png", range_image_range)

    for i, (input, target) in enumerate(dataloader):
        input['proj'] = input['proj'].to(device)
        input['proj_mask'] = input['proj_mask'].to(device)
        target['proj'] = target['proj'].to(device)
        target['proj_mask'] = target['proj_mask'].to(device)

        torch.cuda.synchronize()  # Wait for all kernels to finish

        data_time = time.time() - start

        start = time.time()

        optimizer.zero_grad()  # Clear the gradients

        # Forward pass
        out = model(input)

        loss = objective(out, target)  # Compute the loss

        # Backward pass
        loss.backward()

        optimizer.step()  # Update the parameters
        out[:, 0].data *= 1.6433 + 2.1008
        out[:, 1].data *= 6.2918 + 6.1954
        out[:, 2].data *= 7.0150 + 6.9538
        out[:, 3].data *= 4.9339 + 12.7296
        out[:, 4].data *= 0.6881 + 0.3313
        out[:, 0].data -= 1.6433
        out[:, 1].data -= 6.2918
        out[:, 2].data -= 7.0150
        out[:, 3].data -= 4.9339
        out[:, 4].data -= 0.6881
        target['proj'][:, 0] *= 1.6433 + 2.1008
        target['proj'][:, 1] *= 6.2918 + 6.1954
        target['proj'][:, 2] *= 7.0150 + 6.9538
        target['proj'][:, 3] *= 4.9339 + 12.7296
        target['proj'][:, 4] *= 0.6881 + 0.3313
        target['proj'][:, 0] -= 1.6433
        target['proj'][:, 1] -= 6.2918
        target['proj'][:, 2] -= 7.0150
        target['proj'][:, 3] -= 4.9339
        target['proj'][:, 4] -= 0.6881

        gpu_time = time.time() - start

        # Calculate Error metrics
        err = create_error_metric(args)
        # err.evaluate(out[:, :5, :, :].data, target['proj'].data)
        err.evaluate(out[:, :5, :, :].data * target['proj_mask'].unsqueeze(1), (target['proj'] * target['proj_mask'].unsqueeze(1)).data)
        # err.evaluate(out[:, :1, :, :].data, (target['proj'][:, :1] * target['proj_mask'].unsqueeze(1)).data)
        err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input['proj'].size(0))

        if (i + 1) % args.print_freq == 0 or i == len(dataloader)-1:
            print('[Train] Epoch ({}) [{}/{}]: '.format(
                epoch, i+1, len(dataloader)),  end='')
            print(err_avg)

        # Log to Tensorboard if enabled
        if tb_writer is not None:
            if (i + 1) % tb_freq == 0:
                avg_meter = err_avg.get_avg()
                tb_writer.add_scalar('Loss/train', avg_meter.loss, epoch * len(dataloader) + i)
                tb_writer.add_scalar('MAE/train', avg_meter.metrics['mae'], epoch * len(dataloader) + i)
                tb_writer.add_scalar('RMSE/train', avg_meter.metrics['rmse'], epoch * len(dataloader) + i)

        start = time.time()  # Start counting again for the next iteration

    return err_avg


############ EVALUATION FUNCTION ############
def evaluate_epoch(dataloader, model, objective, epoch, train=False):
    """
    Evluation function
    
    Args:
        dataloader: The dataloader object for the dataset
        model: The model to be trained
        epoch: What epoch to start from
    
    Returns:
        AverageMeter() object.
    
    Raises:
        KeyError: Raises an exception.
    """
    print('\n==> Evaluating Epoch [{}]'.format(epoch))
    if train:
        if not os.path.exists(os.path.join('pncnn_output', 'train')):
            os.makedirs(os.path.join('pncnn_output', 'train'))
            os.makedirs(os.path.join('pncnn_output', 'train', 'nusc_velodyne'))
            os.makedirs(os.path.join('pncnn_output', 'train', 'kitti_velodyne'))
    else:
        if not os.path.exists(os.path.join('pncnn_output', 'valid')):
            os.makedirs(os.path.join('pncnn_output', 'valid'))
            os.makedirs(os.path.join('pncnn_output', 'valid', 'nusc_velodyne'))
            os.makedirs(os.path.join('pncnn_output', 'valid', 'kitti_velodyne'))
    if train:
        if not os.path.exists(os.path.join('pncnn_input', 'train')):
            os.makedirs(os.path.join('pncnn_input', 'train'))
            os.makedirs(os.path.join('pncnn_input', 'train', 'nusc_velodyne'))
            os.makedirs(os.path.join('pncnn_input', 'train', 'kitti_velodyne'))
    else:
        if not os.path.exists(os.path.join('pncnn_input', 'valid')):
            os.makedirs(os.path.join('pncnn_input', 'valid'))
            os.makedirs(os.path.join('pncnn_input', 'valid', 'nusc_velodyne'))
            os.makedirs(os.path.join('pncnn_input', 'valid', 'kitti_velodyne'))


    err = create_error_metric(args)
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model.eval()  # Swith to evaluate mode

    # Save output images
    out_img_saver = create_out_image_saver(exp_dir, args, epoch)
    out_image = None

    if not os.path.exists('imgs/'):
        os.makedirs('imgs/')

    start = time.time()
    with torch.no_grad(): # Disable gradients computations
        # visualizations of range image
        # for i, (input, target) in enumerate(dataloader):
        #         # plot
        #         range_image_range = input['proj'][0,0].numpy()
        #         range_image_x = input['proj'][0,1].numpy()
        #         range_image_y = input['proj'][0,2].numpy()
        #         range_image_z = input['proj'][0,3].numpy()
        #         range_image_remission = input['proj'][0,4].numpy()
        #         range_image_range = (range_image_range / range_image_range.max() * 255.).astype('int')
        #         Image.imsave('imgs/'+str(i)+'_'+"kitti_lidar.png", range_image_range)
        #
        #         range_image_range = target['proj'][0,0].numpy()
        #         range_image_x = target['proj'][0,1].numpy()
        #         range_image_y = target['proj'][0,2].numpy()
        #         range_image_z = target['proj'][0,3].numpy()
        #         range_image_remission = target['proj'][0,4].numpy()
        #         range_image_range = (range_image_range / range_image_range.max() * 255.).astype('int')
        #         Image.imsave('imgs/'+str(i)+'_'+"high_res_lidar.png", range_image_range)

        for i, (input, target) in enumerate(dataloader):
            # plot
            range_image_range = input['proj'][0,0].numpy()
            range_image_x = input['proj'][0,1].numpy()
            range_image_y = input['proj'][0,2].numpy()
            range_image_z = input['proj'][0,3].numpy()
            range_image_remission = input['proj'][0,4].numpy()
            range_image_range = (range_image_range / range_image_range.max() * 255.).astype('int')
            Image.imsave('imgs/'+str(i)+'_'+"kitti_lidar.png", range_image_range)

            range_image_range = target['proj'][0,0].numpy()
            range_image_x = target['proj'][0,1].numpy()
            range_image_y = target['proj'][0,2].numpy()
            range_image_z = target['proj'][0,3].numpy()
            range_image_remission = target['proj'][0,4].numpy()
            range_image_range = (range_image_range / range_image_range.max() * 255.).astype('int')
            Image.imsave('imgs/'+str(i)+'_'+"high_res_lidar.png", range_image_range)

            input['proj'] = input['proj'].to(device)
            input['proj_mask'] = input['proj_mask'].to(device)
            target['proj'] = target['proj'].to(device)
            target['proj_mask'] = target['proj_mask'].to(device)

            torch.cuda.synchronize()

            data_time = time.time() - start

            # Forward Pass
            start = time.time()

            out = model(input)
            range_image_range = (out.detach().cpu().numpy()[0,0] / out.detach().cpu().numpy()[0,0].max() * 255.).astype('int')
            Image.imsave('imgs/'+str(i)+'_'+"high_res_lidar_pred.png", range_image_range)

            # Check if there is cout There is Cout
            loss = objective(out, target)  # Compute the loss
            out[:, 0].data *= 1.6433 + 2.1008
            out[:, 1].data *= 6.2918 + 6.1954
            out[:, 2].data *= 7.0150 + 6.9538
            out[:, 3].data *= 4.9339 + 12.7296
            out[:, 4].data *= 0.6881 + 0.3313
            out[:, 0].data -= 1.6433
            out[:, 1].data -= 6.2918
            out[:, 2].data -= 7.0150
            out[:, 3].data -= 4.9339
            out[:, 4].data -= 0.6881
            target['proj'][:, 0] *= 1.6433 + 2.1008
            target['proj'][:, 1] *= 6.2918 + 6.1954
            target['proj'][:, 2] *= 7.0150 + 6.9538
            target['proj'][:, 3] *= 4.9339 + 12.7296
            target['proj'][:, 4] *= 0.6881 + 0.3313
            target['proj'][:, 0] -= 1.6433
            target['proj'][:, 1] -= 6.2918
            target['proj'][:, 2] -= 7.0150
            target['proj'][:, 3] -= 4.9339
            target['proj'][:, 4] -= 0.6881


            # if train:
            #     np.save(os.path.join('pncnn_output', 'train', input['depth_path'][0].split('/')[-2], input['depth_path'][0].split('/')[-1].replace('bin', 'npy')), out[:,:5].cpu().detach().numpy())
            # else:
            #     np.save(os.path.join('pncnn_output', 'valid', input['depth_path'][0].split('/')[-2], input['depth_path'][0].split('/')[-1].replace('bin', 'npy')), out[:,:5].cpu().detach().numpy())
            if train:
                np.save(os.path.join('pncnn_input', 'train', input['depth_path'][0].split('/')[-2], input['depth_path'][0].split('/')[-1].replace('bin', 'npy')), input['proj'][:,:5].cpu().detach().numpy())
            else:
                np.save(os.path.join('pncnn_input', 'valid', input['depth_path'][0].split('/')[-2], input['depth_path'][0].split('/')[-1].replace('bin', 'npy')), input['proj'][:,:5].cpu().detach().numpy())

            gpu_time = time.time() - start

            # Calculate Error metrics
            err = create_error_metric(args)
            # err.evaluate(out[:, :5, :, :].data, target['proj'].data)
            err.evaluate(out[:, :1, :, :].data * target['proj_mask'].unsqueeze(1), (target['proj'][:, :1] * target['proj_mask'].unsqueeze(1)).data)
            err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input['proj'].size(0))

            # Save output images
            if args.save_val_imgs: # richard
                out_image = out_img_saver.update(i, out_image, input, out, target)

            if args.evaluate is None:
                if tb_writer is not None and i == 1:  # Retrun batch 1 for tensorboard logging
                    out_image = out

            if (i + 1) % args.print_freq == 0 or i == len(dataloader)-1:
                print('[Eval] Epoch: ({0}) [{1}/{2}]: '.format(
                    epoch, i + 1, len(dataloader)), end='')
                print(err_avg)

            start = time.time()

    return err_avg, out_image


if __name__ == '__main__':
    main()
