"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    @staticmethod
    def _extract_metric_value(stats, key: str, index: int = 0):
        if not stats or key not in stats:
            return None
        value = stats[key]
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            if index < 0 or index >= len(value):
                return None
            return float(value[index])
        try:
            return float(value)
        except Exception:
            return None
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        eval_freq = getattr(args, 'eval_freq', 1)
        save_last_freq = getattr(args, 'save_last_freq', 1)

        early_stop_patience = getattr(args, 'early_stop_patience', 0)
        early_stop_metric = getattr(args, 'early_stop_metric', 'coco_eval_bbox')
        early_stop_index = getattr(args, 'early_stop_index', 0)
        early_stop_mode = getattr(args, 'early_stop_mode', 'max')
        early_stop_min_delta = getattr(args, 'early_stop_min_delta', 0.0)

        best_es_value = None
        bad_eval_count = 0

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = []

                if save_last_freq and save_last_freq > 0:
                    if (epoch + 1) % save_last_freq == 0 or (epoch + 1) == args.epoches:
                        checkpoint_paths.append(self.output_dir / 'last.pth')

                checkpoint_freq = getattr(args, 'checkpoint_freq', 0)
                if checkpoint_freq and checkpoint_freq > 0:
                    if (epoch + 1) % checkpoint_freq == 0:
                        checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            test_stats = {}
            coco_evaluator = None

            do_eval = False
            if eval_freq and eval_freq > 0:
                do_eval = (epoch + 1) % eval_freq == 0 or (epoch + 1) == args.epoches

            if do_eval:
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                )

                if early_stop_patience and early_stop_patience > 0:
                    current_value = self._extract_metric_value(test_stats, early_stop_metric, early_stop_index)
                    if current_value is not None:
                        if best_es_value is None:
                            best_es_value = current_value
                            bad_eval_count = 0
                        else:
                            if early_stop_mode == 'min':
                                improved = current_value < (best_es_value - float(early_stop_min_delta))
                            else:
                                improved = current_value > (best_es_value + float(early_stop_min_delta))

                            if improved:
                                best_es_value = current_value
                                bad_eval_count = 0
                            else:
                                bad_eval_count += 1

            # synchronize early stop across ranks
            stop_training = False
            if do_eval and early_stop_patience and early_stop_patience > 0 and best_es_value is not None:
                if bad_eval_count >= int(early_stop_patience):
                    stop_training = True

            if dist_utils.is_dist_available_and_initialized():
                stop_flag = torch.tensor(int(stop_training), device=self.device)
                torch.distributed.broadcast(stop_flag, src=0)
                stop_training = bool(stop_flag.item())

            if stop_training:
                if dist_utils.is_main_process():
                    print(
                        f'Early stopping at epoch {epoch}: '
                        f'metric={early_stop_metric}[{early_stop_index}] best={best_es_value} '
                        f'bad_evals={bad_eval_count} patience={early_stop_patience}'
                    )
                if self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')
                break

            if do_eval:
                # TODO 
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]

                    if best_stat['epoch'] == epoch and self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

                print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        if self.output_dir and dist_utils.is_main_process():
            print(f'Last checkpoint: {self.output_dir / "last.pth"}')
            print(f'Best checkpoint: {self.output_dir / "best.pth"}')


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
