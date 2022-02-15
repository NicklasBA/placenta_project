#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import sys
sys.path.append(r"C:\Users\ptrkm\PycharmProjects\placenta_project\SlowFast")

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job, write_results
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize
import os
import sys
import torch

def eval_net(cfg):
    """
    Function to spawn the evaluation a network
    """
    # cfg = load_config(args)
    # cfg = assert_and_infer_cfg(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        results = launch_job(cfg=cfg, init_method=cfg.init_method, func=test)
    else:
        print("test must be enabled for this function, for train call run_net")


    write_results(results, cfg)
    print(f"Results were saved onto {cfg.OUTPUT_DIR}")
    # Perform model visualization.






