if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from Manip_Flow.workspace.base_workspace import BaseWorkspace
from Manip_Flow.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from Manip_Flow.dataset.base_dataset import BaseImageDataset, BaseDataset
from Manip_Flow.env_runner.base_image_runner import BaseImageRunner
from Manip_Flow.common.checkpoint_util import TopKCheckpointManager
from Manip_Flow.common.json_logger import JsonLogger, NullJsonLogger
from Manip_Flow.common.pytorch_util import dict_apply, optimizer_to
from Manip_Flow.common.optim_groups import build_param_groups
from Manip_Flow.common.val_diagnostics import (
    log_draw_dispersion,
    log_prefix_consistency,
    shuffled_obs_batch,
)
from Manip_Flow.model.diffusion.ema_model import EMAModel
from Manip_Flow.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        # Split the obs_encoder into the PRETRAINED backbone (obs_encoder.key_model_map,
        # i.e. the timm ViT/ResNet weights) and the feature-aggregation heads
        # (attention_pool_2d / spatial_embedding / aggregation_transformer / attention),
        # which are randomly initialized. Only the former wants the reduced lr --
        # giving a from-scratch head 0.1x lr just makes it converge slowly.
        backbone_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder.pretrained:
            backbone_lr *= 0.1
            print('==> reduce pretrained obs_encorder backbone lr')
        param_groups = build_param_groups(self.model, backbone_lr)
        labels = ['velocity model', 'obs_encorder backbone', 'obs_encorder head',
                  'other policy']
        for label, group in zip(labels, param_groups):
            print(f'{label} params: {len(group["params"])} '
                  f'@ lr {group.get("lr", cfg.optimizer.lr):g}')
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        # Only rank 0 may write logs.json.txt: all ranks share output_dir, and
        # concurrent line-buffered appends interleave mid-record, corrupting the
        # file so that the next run crashes in JsonLogger.start()/json.loads.
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger_ctx = JsonLogger(log_path) if accelerator.is_main_process \
            else NullJsonLogger()
        with json_logger_ctx as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                # NOTE: val_dataloader went through accelerator.prepare, so it is
                # SHARDED per rank -- every rank must iterate its own shard and the
                # result be reduced, otherwise main-process-only validation would
                # only ever see 1/world_size of the val episodes. Running the forward
                # on all ranks also keeps DDP participation symmetric.
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    # val_loss is measured on `policy`, i.e. the EMA copy when
                    # use_ema=True -- those are exactly the weights that get
                    # deployed (inference.py prefers state_dicts['ema_model']).
                    # topk ranks checkpoints by val_loss, so measuring it on the
                    # raw model selects a checkpoint by a number that never runs.
                    # val_loss_raw keeps the pre-2026-08-08 series comparable.
                    #
                    # `policy.eval()` above only touches ema_model when use_ema=True,
                    # so put the trained model itself in eval explicitly -- otherwise
                    # RandomCrop/ColorJitter stay active and val_loss is noise.
                    # self.model.train() at the top of the next epoch restores mode.
                    # Both models therefore see UNAUGMENTED inputs while train_loss
                    # is augmented, so the train/val gap logged here UNDERSTATES
                    # overfitting.
                    #
                    # val_loss_shuffled_obs is the same flow loss with each obs
                    # paired to another sample's action. The flow target x1-x0 has
                    # an irreducible conditional-variance floor, so the absolute
                    # val_loss carries no scale on its own; this is the
                    # "conditioning is uninformative" reference. val_loss rising
                    # toward it means the visual conditioning has stopped paying.
                    self.model.eval()
                    run_val_diagnostics = bool(OmegaConf.select(
                        cfg, 'training.val_diagnostics', default=True))
                    with torch.no_grad():
                        # [deployed(ema), raw, shuffled-obs, shuffled batch count]
                        val_sums = torch.zeros(4, device=device)
                        val_batches = torch.zeros((), device=device)
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec,
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                val_sums[0] += policy(batch).detach().float()
                                if run_val_diagnostics:
                                    # With use_ema=False `policy` already IS the
                                    # unwrapped self.model, so skip the duplicate pass.
                                    if cfg.training.use_ema:
                                        val_sums[1] += self.model(batch).detach().float()
                                    else:
                                        val_sums[1] += val_sums[0].detach()
                                    if batch['action'].shape[0] > 1:
                                        val_sums[2] += policy(
                                            shuffled_obs_batch(batch)).detach().float()
                                        val_sums[3] += 1
                                val_batches += 1
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        # weight by per-rank batch count: shards can be uneven
                        stats = accelerator.gather(
                            torch.cat([val_sums, val_batches.reshape(1)]).unsqueeze(0))
                        totals = stats.sum(dim=0)
                        total_batches = totals[4].item()
                        if total_batches > 0:
                            # log epoch average validation loss
                            step_log['val_loss'] = totals[0].item() / total_batches
                            if run_val_diagnostics:
                                step_log['val_loss_raw'] = totals[1].item() / total_batches
                                shuffled_batches = totals[3].item()
                                if shuffled_batches > 0:
                                    step_log['val_loss_shuffled_obs'] = \
                                        totals[2].item() / shuffled_batches


                # NOTE: values MUST be python floats, not 0-dim tensors. JsonLogger's
                # default filter_fn drops anything that is not a numbers.Number
                # (json_logger.py:70-72), and torch.Tensor does not register with the
                # numbers ABC -- so tensor-valued entries were silently discarded and
                # never reached logs.json.txt (every run before 2026-08-07 logged only
                # train_loss/val_loss/lr, despite sample_every firing on schedule).
                def log_action_mse(step_log, category, pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, 10)
                    gt_action = gt_action.view(B, T, -1, 10)
                    mse = torch.nn.functional.mse_loss
                    step_log[f'{category}_action_mse_error'] = mse(pred_action, gt_action).item()
                    step_log[f'{category}_action_mse_error_pos'] = mse(pred_action[..., :3], gt_action[..., :3]).item()
                    step_log[f'{category}_action_mse_error_rot'] = mse(pred_action[..., 3:9], gt_action[..., 3:9]).item()
                    step_log[f'{category}_action_mse_error_width'] = mse(pred_action[..., 9], gt_action[..., 9]).item()

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                        log_action_mse(step_log, 'train', pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                            log_action_mse(step_log, 'val', pred_action, gt_action)
                            if bool(OmegaConf.select(
                                    cfg, 'training.val_diagnostics', default=True)):
                                n_obs = min(
                                    int(OmegaConf.select(
                                        cfg, 'training.val_draw_n_obs', default=4)),
                                    gt_action.shape[0])
                                log_draw_dispersion(
                                    step_log, policy, batch['obs'], n_obs,
                                    int(OmegaConf.select(
                                        cfg, 'training.val_draw_k', default=8)))
                                # deploy's stride=32 advances ~11 of the 40 action
                                # tokens per commit, i.e. about half the RTC
                                # execution horizon.
                                log_prefix_consistency(
                                    step_log, policy, batch['obs'], n_obs,
                                    policy.rtc_execution_horizon // 2)

                        del batch
                        del gt_action
                        del pred_action
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # topk monitors val_loss, which is absent if this epoch ran no
                    # validation (val_every not dividing checkpoint_every, or an
                    # empty val split). Skip topk instead of dying on a KeyError --
                    # save_last_ckpt above has already persisted this epoch.
                    if cfg.checkpoint.topk.monitor_key not in metric_dict:
                        print(f"[topk] epoch {self.epoch}: no "
                              f"'{cfg.checkpoint.topk.monitor_key}' in metrics, "
                              f"skipping topk checkpoint")
                    else:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # Do NOT call accelerator.end_training() here. In accelerate >= 1.x it ends
        # trackers AND calls PartialState().destroy_process_group(). Under Ray Train
        # the controller then runs its own _shutdown_torch() ->
        # dist.destroy_process_group(), which trips `assert pg is not None` and makes
        # md_ai_kit mark an otherwise-COMPLETED run FAILED (2026-08-07,
        # flow_humi_unet_dino_drawer_a40_ep12: all 12 epochs + all checkpoints
        # written, then the job reported FAILED on teardown).
        # Whoever created the process group should destroy it, so only do the
        # tracker half of end_training() here.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            for tracker in getattr(accelerator, 'trackers', []):
                tracker.finish()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
