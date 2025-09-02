"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # Initialize wandb if enabled and on main process
    wandb_run = None
    if _WANDB_AVAILABLE and args.use_wandb and dist_utils.is_main_process():
        # Override wandb enabled setting from CLI
        if not cfg.wandb:
            cfg.wandb = {}
        cfg.wandb['enabled'] = True
        if args.wandb_id:
            cfg.wandb['id'] = args.wandb_id
        
        wandb_run = wandb.init(
            project=cfg.wandb.get('project', 'DEIM-detection'),
            entity=cfg.wandb.get('entity', None),
            name=cfg.wandb.get('name', None),
            id=cfg.wandb.get('id', None),
            config=cfg.yaml_cfg,
            resume='auto' if cfg.wandb.get('id') else False,
            reinit=True,
            tags=cfg.wandb.get('tags', []),
            notes=cfg.wandb.get('notes', None),
            group=cfg.wandb.get('group', None),
            job_type=cfg.wandb.get('job_type', 'training'),
            anonymous=cfg.wandb.get('anonymous', None),
            mode=cfg.wandb.get('mode', 'online'),
            save_code=cfg.wandb.get('save_code', True),
        )
        
        cfg.wandb_run = wandb_run
        print("WandB initialized successfully!")
    elif args.use_wandb and not _WANDB_AVAILABLE:
        print("Warning: --use-wandb specified but wandb is not available. Please install wandb.")
    elif args.use_wandb and not dist_utils.is_main_process():
        print("WandB will only log from main process in distributed training.")

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # wandb arguments
    parser.add_argument('--use-wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--wandb-id', type=str, help='resume existing run id')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
