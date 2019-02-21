# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/21 20:28
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import glob
import os
import shutil
import socket
from datetime import datetime
from tensorboardX import SummaryWriter


def get_save_path(args, check=False):
    save_dir_root = os.getcwd()
    save_dir_root = os.path.join(save_dir_root, 'result', args.model)
    if args.resume:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        if len(runs) > 10:
            print('please delete unnecessary runs, ensure run_id < 10.')
        if check:
            run_id = int(runs[-1].split('_')[-1]) if runs else 0
        else:
            run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_logger(output_directory):
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)
    return logger


def write_config_file(args, output_directory):
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)