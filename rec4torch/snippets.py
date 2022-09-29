import numpy as np
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import time
import sys
import collections
import inspect
from datetime import datetime
import warnings


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05, stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class Callback(object):
    '''Callback基类
    '''
    def __init__(self):
        pass
    def on_train_begin(self, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass
    def on_epoch_begin(self, global_step, epoch, logs=None):
        pass
    def on_epoch_end(self, global_step, epoch, logs=None):
        pass
    def on_batch_begin(self, global_step, local_step, logs=None):
        pass
    def on_batch_end(self, global_step, local_step, logs=None):
        pass
    def on_dataloader_end(self, logs=None):
        pass


class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.

    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).

    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, epochs, steps, metrics, stateful_metrics=None, verbose=1):
        super(ProgbarLogger, self).__init__()
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()
        self.params = {'epochs': epochs, 'steps': steps, 'verbose': verbose, 'metrics': metrics}
        self.verbose = verbose
        self.epochs = epochs

    def add_metrics(self, metrics, stateful_metrics=None, add_position=None):
        if add_position is None:
            add_position = len(self.params['metrics'])
        metrics = [metrics] if isinstance(metrics, str) else metrics
        if stateful_metrics:
            stateful_metrics = [stateful_metrics] if isinstance(stateful_metrics, str) else stateful_metrics
            self.stateful_metrics.update(set(stateful_metrics))
            self.progbar.stateful_metrics.update(set(stateful_metrics))

        add_metrics = []
        for metric in metrics:
            if metric not in self.params['metrics']:
                add_metrics.append(metric)
        self.params['metrics'] = self.params['metrics'][:add_position] + add_metrics + self.params['metrics'][add_position:]

    def on_train_begin(self, logs=None):
        if self.verbose:
            print('Start Training'.center(40, '='))

    def on_epoch_begin(self, global_step=None, epoch=None, logs=None):
        if self.verbose:
            time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s - Epoch: %d/%d' % (time_start, epoch + 1, self.epochs))
            self.target = self.params['steps']
            self.progbar = Progbar(target=self.target, verbose=self.verbose, stateful_metrics=self.stateful_metrics)
        self.seen = 0

    def on_batch_begin(self, global_step=None, local_step=None, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, global_step=None, local_step=None, logs=None):
        logs = logs or {}
        self.seen += 1
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, global_step=None, epoch=None, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)
    
    def on_train_end(self, logs=None):
        if self.verbose:
            print('Finish Training'.center(40, '='))


class EarlyStopping(Callback):
    '''Stop training策略, 从keras中移植
    '''
    def __init__(self, monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, fallback to auto mode.' % mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.greater if 'acc' in self.monitor else np.less
        self.min_delta = self.min_delta if self.monitor_op == np.greater else -self.min_delta

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, steps, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch+1}: early stopping\n')

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn('Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
        return monitor_value


class Logger(Callback):
    '''默认logging
    对于valid/dev和test的日志需要在evaluate之后对log进行赋值，如log['dev_f1']=f1，并在Evaluator之后调用
    若每隔一定steps对验证集评估，则Logger的interval设置成和Evaluater一致或者约数，保证日志能记录到
    '''
    def __init__(self, filename, interval=10, verbosity=1, name=None):
        super(Logger, self).__init__()
        self.interval = interval

        import logging
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level_dict[verbosity])
        fh = logging.FileHandler(filename, "a")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def on_train_begin(self, logs=None):
        self.logger.info('Start Training'.center(40, '='))

    def on_train_end(self, logs=None):
        self.logger.info('Finish Training'.center(40, '='))

    def on_epoch_begin(self, global_step, epoch, logs=None):
        self.logger.info(f'Epoch {epoch}'.center(40, '='))

    def on_epoch_end(self, global_step, epoch, logs=None):
        log_str = '\t '.join([f'{k}={v:.5f}' for k, v in logs.items()])
        self.logger.info(f'epoch={epoch+1}\t {log_str}')

    def on_batch_end(self, global_step, local_step, logs=None):
        if (global_step+1) % self.interval == 0:
            log_str = '\t '.join([f'{k}={v:.5f}' for k, v in logs.items()])
            self.logger.info(f'step={global_step+1}\t {log_str}')


class Tensorboard(Callback):
    '''默认Tensorboard
    对于valid/dev和test的Tensorboard需要在evaluate之后对log进行赋值，如log['dev/f1']=f1，并在Evaluator之后调用
    赋值需要分栏目的用'/'进行分隔
    若每隔一定steps对验证集评估，则Tensorboard的interval设置成和Evaluater一致或者约数，保证Tensorboard能记录到
    '''
    def __init__(self, dirname, interval=10, prefix='train'):
        super(Tensorboard, self).__init__()
        self.interval = interval
        self.prefix = prefix

        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=str(dirname))  # prepare summary writer

    def on_epoch_end(self, global_step, epoch, logs=None):
        for k, v in logs.items():
            index = k if '/' in k else f"{self.prefix}/{k}"
            self.writer.add_scalar(index, v, global_step)

    def on_batch_end(self, global_step, local_step, logs=None):
        if (global_step+1) % self.interval == 0:
            for k, v in logs.items():
                index = k if '/' in k else f"{self.prefix}/{k}"
                self.writer.add_scalar(index, v, global_step)


def metric_mapping(metric, func, y_pred, y_true):
    # 自定义metrics
    if inspect.isfunction(func):
        metric_res = func(y_pred, y_true)
        if inspect.isfunction(metric):
            # 如果直接传入回调函数（无key），要求回调函数返回Dict[String: Int/Float]类型
            assert isinstance(metric_res, dict), 'Custom metrics callbacks should return "Dict[String: Int/Float]" value'
        elif isinstance(metric, str):
            # 如果直接传入回调函数（有key），要求回调函数返回Int/Float类型
            assert isinstance(metric_res, (int, float)), 'Custom metrics callbacks should return "Int, Float" value'
        return metric_res
    elif metric == 'loss':
        pass
    # 自带metrics
    elif isinstance(metric, str):
        # 如果forward返回了list, tuple，则选取第一项
        y_pred_tmp = y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred
        y_true_tmp = y_true[0] if isinstance(y_true, (list, tuple)) else y_true

        # 根据shape做预处理
        if len(y_pred_tmp.shape) == len(y_true_tmp.shape) + 1:
            y_pred_tmp = torch.argmax(y_pred_tmp, dim=-1)
        elif len(y_pred_tmp.shape) == len(y_true_tmp.shape):
            pass
        else:
            raise ValueError(f'y_pred_tmp.shape={y_pred_tmp.shape} while y_true_tmp.shape={y_true_tmp.shape}')

        # 执行内置的metric
        if metric in {'accuracy', 'acc'}:
            return torch.sum(y_pred_tmp.eq(y_true_tmp)).item() / y_true_tmp.numel()
        elif metric in {'mae', 'MAE', 'mean_absolute_error'}:
            return torch.mean(torch.abs(y_pred_tmp - y_true_tmp)).item()
        elif metric in {'mse', 'MSE', 'mean_squared_error'}:
            return torch.mean(torch.square(y_pred_tmp - y_true_tmp)).item()
        elif metric in {'mape', 'MAPE', 'mean_absolute_percentage_error'}:
            diff = torch.abs((y_true_tmp - y_pred_tmp) / torch.clamp(torch.abs(y_true_tmp), 1e-7, None))
            return 100. * torch.mean(diff).item()
        elif metric in {'msle', 'MSLE', 'mean_squared_logarithmic_error'}:
            first_log = torch.log(torch.clamp(y_pred_tmp, 1e-7, None) + 1.)
            second_log = torch.log(torch.clamp(y_true_tmp, 1e-7, None) + 1.)
            return torch.mean(torch.square(first_log - second_log)).item()

    return None