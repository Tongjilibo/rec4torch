import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import time
from collections import OrderedDict
from inspect import isfunction
from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from ..snippets import metric_mapping, ProgbarLogger, EarlyStopping
from ..layers import PredictionLayer


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # 这里主要是为了外面调用用到
        self.global_step, self.local_step, self.total_steps, self.epoch, self.steps_per_epoch, self.train_dataloader = 0, 0, 0, 0, None, None
        self.resume_step, self.resume_epoch = 0, 0
        self.callbacks = []
    
    def save_steps_params(self, save_path):
        '''保存训练过程参数
        '''
        step_params = {'resume_step': (self.local_step+1) % self.steps_per_epoch, 
                       'resume_epoch': self.epoch + (self.local_step+1) // self.steps_per_epoch}
        torch.save(step_params, save_path)

    def load_steps_params(self, save_path):
        '''导入训练过程参数
        '''
        step_params = torch.load(save_path)
        self.resume_step = step_params['resume_step'] 
        self.resume_epoch = step_params['resume_epoch']
        return step_params

    def compile(self, loss, optimizer, scheduler=None, clip_grad_norm=None, use_amp=False, metrics=None, adversarial_train={'name': ''}):
        '''定义loss, optimizer, metrics, 是否在计算loss前reshape
        loss: loss
        optimizer: 优化器
        scheduler: scheduler
        clip_grad_norm: 是否使用梯度裁剪, 默认不启用
        use_amp: 是否使用混合精度，默认不启用
        metrics: 训练过程中需要打印的指标, loss相关指标默认会打印, 目前支持accuracy, 也支持自定义metric，形式为{key: func}
        '''
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.use_amp = use_amp
        if use_amp:
            assert adversarial_train['name'] not in {'vat', 'gradient_penalty'}, 'Amp and adversarial_train both run is not supported in current version'
            from torch.cuda.amp import autocast
            self.autocast = autocast
            self.scaler = torch.cuda.amp.GradScaler()

        # 训练过程观测的指标
        self.metrics = OrderedDict({'loss': None})
        if metrics is None:
            metrics = []
        elif isinstance(metrics, (str, dict)) or isfunction(metrics):
            metrics = [metrics]

        for metric in metrics:
            # 字符类型，目前仅支持accuracy
            if isinstance(metric, str) and metric != 'loss':
                self.metrics[metric] = None
            # 字典形式 {metric: func}
            elif isinstance(metric, dict):
                self.metrics.update(metric)
            # 函数形式，key和value都赋值metric
            elif isfunction(metric):
                self.metrics.update({metric: metric})
            else:
                raise ValueError('Args metrics only support "String, Dict, Callback, List[String, Dict, Callback]" format')

    def train_step(self, train_X, train_y, grad_accumulation_steps):
        '''forward并返回loss
        '''
        def args_segmentate(train_X):
            '''参数是否展开
            '''
            if isinstance(train_X, torch.Tensor):  # tensor不展开
                pass
            elif isinstance(self, (BaseModelDP, BaseModelDDP)):
                if self.module.forward.__code__.co_argcount >= 3:
                    return True
            elif self.forward.__code__.co_argcount >= 3:
                return True
            return False

        if self.use_amp:
            with self.autocast():
                output = self.forward(*train_X) if args_segmentate(train_X) else self.forward(train_X)
                loss_detail = self.criterion(output, train_y)
        else:
            output = self.forward(*train_X) if args_segmentate(train_X) else self.forward(train_X)
            loss_detail = self.criterion(output, train_y)

        if isinstance(loss_detail, torch.Tensor):
            loss = loss_detail
            loss_detail = {}
        elif isinstance(loss_detail, dict):
            loss = loss_detail['loss']  # 还存在其他loss，仅用于打印
            del loss_detail['loss']
        elif isinstance(loss_detail, (tuple, list)):
            loss = loss_detail[0]
            loss_detail = {f'loss{i}':v for i, v in enumerate(loss_detail[1:], start=1)}
        else:
            raise ValueError('Return loss only support Tensor/dict/tuple/list format')
        # 梯度累积
        loss = loss / grad_accumulation_steps if grad_accumulation_steps > 1 else loss
        return output, loss, loss_detail

    def callback_fun(self, mode, logs={}):
        '''统一调用callback, 方便一些判断条件的触发
        '''
        # 如果是分布式DDP训练，则仅masker_rank可以callback
        if isinstance(self, BaseModelDDP) and self.master_rank!=torch.distributed.get_rank():
            return

        if mode == 'train_begin':
            for callback in self.callbacks:
                callback.on_train_begin()
        elif mode == 'epoch_begin':
            for callback in self.callbacks:
                callback.on_epoch_begin(self.global_step, self.epoch, logs)
        elif mode == 'batch_begin':
            for callback in self.callbacks:
                callback.on_batch_begin(self.global_step, self.local_step, logs)
        elif mode == 'batch_end':
            for callback in self.callbacks:
                callback.on_batch_end(self.global_step, self.local_step, logs)
        elif mode == 'epoch_end':
            for callback in self.callbacks:
                callback.on_epoch_end(self.global_step, self.epoch, logs)
        elif mode == 'train_end':
            for callback in self.callbacks:
                callback.on_train_end()
        elif mode == 'dataloader_end':
            for callback in self.callbacks:
                callback.on_dataloader_end()

    def fit(self, train_dataloader, steps_per_epoch=None, epochs=1, grad_accumulation_steps=1, callbacks=None):
        if not hasattr(train_dataloader, '__len__'):
            assert steps_per_epoch is not None, 'Either train_dataloader has attr "__len__" or steps_per_epoch is not None'

        self.steps_per_epoch = len(train_dataloader) if steps_per_epoch is None else steps_per_epoch
        self.total_steps = self.steps_per_epoch * epochs
        self.train_dataloader = train_dataloader  # 设置为成员变量，可由外部的callbacks进行修改
        train_dataloader_iter = iter(self.train_dataloader)  # 循环epoch时不重生成

        callbacks = [] if callbacks is None else callbacks
        callbacks = callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]
        self.callbacks = [ProgbarLogger(epochs, self.steps_per_epoch, [i for i in self.metrics.keys() if isinstance(i, str)])] + callbacks
        self.callback_fun('train_begin')

        # epoch：当前epoch
        # global_step：当前全局训练步数
        # local_step: 当前epoch内的训练步数，不同epoch中相同local_step对应的batch数据不一定相同，在steps_per_epoch=None时相同
        # bti：在dataloader中的index，不同epoch中相同的bti对应的batch数据一般相同，除非重新生成dataloader
        self.bti = 0
        for epoch in range(self.resume_epoch, epochs):
            self.epoch = epoch
            # resume_step：判断local_step的起点，以及进度条的起始位置
            resume_step = self.resume_step if epoch==self.resume_epoch else 0
            self.callback_fun('epoch_begin')
            self.callbacks[0].seen = resume_step
            
            for local_step in range(resume_step, self.steps_per_epoch):
                self.local_step = local_step
                self.global_step = self.epoch * self.steps_per_epoch + self.local_step
                # 循环dataloader, 不要试用itertools的cycle，遇到过变量不释放的问题
                try:
                    batch = next(train_dataloader_iter)
                except StopIteration:
                    self.callback_fun('dataloader_end')  # 适用于数据量较大时，动态读取文件并重新生成dataloader的情况，如预训练
                    train_dataloader_iter = iter(self.train_dataloader)  # shuffle=True时候，其实顺序也重新生成了
                    self.bti = 0
                    batch = next(train_dataloader_iter)
                train_X, train_y = batch

                logs = OrderedDict()
                self.callback_fun('batch_begin', logs)

                self.train()  # 设置为train模式
                # 入参个数判断，如果入参>=3表示是多个入参，如果=2则表示是一个入参
                output, loss, loss_detail = self.train_step(train_X, train_y, grad_accumulation_steps)
                
                if self.use_amp:  # 混合精度
                    scale_before_step = self.scaler.get_scale()
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 参数更新, 真实的参数更新次数要除以grad_accumulation_steps，注意调整总的训练步数
                if (self.global_step+1) % grad_accumulation_steps == 0:
                    skip_scheduler = False
                    # 混合精度
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        if self.clip_grad_norm is not None:  # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        skip_scheduler = self.scaler.get_scale() != scale_before_step
                    else:
                        if self.clip_grad_norm is not None:  # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                        self.optimizer.step()

                    self.optimizer.zero_grad()  # 清梯度
                    if (self.scheduler is not None) and not skip_scheduler:
                        if isinstance(self.scheduler, (tuple, list)):
                            for scheduler in self.scheduler:
                                scheduler.step()
                        else:
                            self.scheduler.step()

                # 添加loss至log打印
                logs.update({'loss': loss.item()})
                logs_loss_detail = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_detail.items()}
                logs.update(logs_loss_detail)
                if self.global_step == resume_step:
                    self.callbacks[0].add_metrics(list(logs_loss_detail.keys()), add_position=1)
                    
                # 添加metrics至log打印
                for metric, func in self.metrics.items():
                    perf = metric_mapping(metric, func, output, train_y)  # 内置的一些accuracy指标
                    if perf is not None:
                        if isfunction(metric):  # 直接传入回调函数(无key)
                            if self.global_step == resume_step:
                                self.callbacks[0].add_metrics(list(perf.keys()))
                            logs.update(perf)
                        elif isinstance(metric, str):  # 直接传入回调函数(有key)
                            logs[metric] = perf

                self.callback_fun('batch_end', logs)

                self.bti += 1
            self.callback_fun('epoch_end', logs)
            # earlystop策略
            callback_tmp = [callback_tmp for callback_tmp in self.callbacks if isinstance(callback_tmp, EarlyStopping)]
            if callback_tmp and callback_tmp[0].stopped_epoch > 0:
                break
        self.callback_fun('train_end', logs)

    @torch.no_grad()
    def predict(self, input_tensor_list, return_all=None):
        self.eval()
        if self.forward.__code__.co_argcount >= 3:
            output = self.forward(*input_tensor_list)
        else:
            output = self.forward(input_tensor_list)
        if return_all is None:
            return output
        elif isinstance(output, (tuple, list)) and isinstance(return_all, int) and return_all < len(output):
            return output[return_all]
        else:
            raise ValueError('Return format error')


class BaseModelDP(BaseModel, nn.DataParallel):
    '''DataParallel模式使用多gpu的方法
    '''
    def __init__(self, *args, **kwargs):
        nn.DataParallel.__init__(self, *args, **kwargs)


class BaseModelDDP(BaseModel, nn.parallel.DistributedDataParallel):
    '''DistributedDataParallel模式使用多gpu的方法
    '''
    def __init__(self, *args, master_rank=0, **kwargs):
        self.master_rank = master_rank  # 用于记录打印条的rank
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)


class Linear(nn.Module):
    """把Sparse+VarLenSparse经过embeddingg后，和Dense特征Pooling在一起
    feature_columns: 
    feature_index:
    """
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        self.varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
        
        # 特征embdding字典，{feat_name: nn.Embedding()}
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False)

    def forward(self, X, sparse_feat_refine_weight=None):
        # 离散变量过[embedding_size, 1]的embedding, [(btz,1,1), (btz,1,1), ...]
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
                                 for feat in self.sparse_feature_columns]
        # 连续变量直接取值 [(btz, dense_len), (btz, dense_len)]
        dense_value_list = [X[:, self.feature_index[self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]] for feat in self.dense_feature_columns]

        # 变长离散变量过embdding: {feat_name: (btz, seq_len, 1)}, [(btz,1,1), (btz,1,1), ...]
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index, self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, self.varlen_sparse_feature_columns)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1])
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)  # [btz, 1, feat_cnt]
            if sparse_feat_refine_weight is not None:  # 加权
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit
        
        return linear_logit

class RecBase(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, task='binary'):

        super(RecBase, self).__init__()
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,))
        self.aux_loss = torch.zeros((1,))

        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False)
        self.linear_model = Linear(linear_feature_columns, self.feature_index)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError("DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]
