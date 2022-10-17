from nntplib import NNTPProtocolError
from turtle import forward
import torch
from torch import nn
from collections import OrderedDict
from inspect import isfunction
from rec4torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list
from rec4torch.inputs import combined_dnn_input, create_embedding_matrix, embedding_lookup, maxlen_lookup
from rec4torch.layers import FM, DNN, PredictionLayer, AttentionSequencePoolingLayer, InterestExtractor, InterestEvolving, CrossNet
from rec4torch.snippets import metric_mapping, ProgbarLogger, EarlyStopping, split_columns, get_kw


class BaseModel(nn.Module):
    """Trainer
    """
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

    def compile(self, loss, optimizer, scheduler=None, clip_grad_norm=None, use_amp=False, metrics=None):
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
        
        # l1正则和l2正则
        reg_loss = self.get_regularization_loss()
        loss = loss + reg_loss + self.aux_loss

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
    """浅层线性全连接，也就是Wide&Cross的Wide部分
    步骤：
    1. Sparse特征分别过embedding, 得到多个[btz, 1, 1]
    2. VarLenSparse过embeddingg+pooling后，得到多个[btz, 1, 1]
    3. Dense特征直接取用, 得到多个[btz, dense_len]
    4. Sparse和VarLenSparse进行cat得到[btz, 1, featnum]，再sum_pooling得到[btz, 1]的输出
    5. Dense特征过[dense_len, 1]的全连接得到[btz, 1]的输出
    6. 两者求和得到最后输出
    
    参数：
    feature_columns: 各个特征的[SparseFeat, VarlenSparseFeat, DenseFeat, ...]的列表
    feature_index: 每个特征在输入tensor X中的列的起止
    """
    def __init__(self, feature_columns, feature_index, init_std=1e-4, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns, self.dense_feature_columns, self.varlen_sparse_feature_columns = split_columns(feature_columns)
        
        # 特征embdding字典，{feat_name: nn.Embedding()}
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False)
        
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1))
            nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        # 离散变量过[embedding_size, 1]的embedding, [(btz,1,1), (btz,1,1), ...]
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
                                 for feat in self.sparse_feature_columns]
        # 连续变量直接取值 [(btz, dense_len), (btz, dense_len)]
        dense_value_list = [X[:, self.feature_index[self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]] for feat in self.dense_feature_columns]

        # 变长离散变量过embdding: {feat_name: (btz, seq_len, 1)}, [(btz,1,1), (btz,1,1), ...]
        sequence_embed_dict = embedding_lookup(X, self.embedding_dict, self.feature_index, self.varlen_sparse_feature_columns, return_dict=True)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, self.varlen_sparse_feature_columns)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1], device=X.device)
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
                 init_std=1e-4, task='binary', **kwargs):
        super(RecBase, self).__init__()
        self.dnn_feature_columns = dnn_feature_columns
        self.aux_loss = 0  # 目前只看到dien里面使用

        # feat_name到col_idx的映射, eg: {'age':(0,1),...}
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)

        # 为SparseFeat和VarLenSparseFeat特征创建embedding
        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False)
        self.linear_model = Linear(linear_feature_columns, self.feature_index)

        # l1和l2正则
        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        # 输出层
        self.out = PredictionLayer(task, )

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        """SparseFeat和VarLenSparseFeat生成Embedding，VarLenSparseFeat要过Pooling, DenseFeat直接从X中取用
        """
        sparse_feature_columns, dense_feature_columns, varlen_sparse_feature_columns = split_columns(feature_columns)

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError("DenseFeat is not supported in dnn_feature_columns")

        # 离散特征过embedding
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in sparse_feature_columns]

        # 序列离散特征过embedding+pooling
        sequence_embed_dict = embedding_lookup(X, self.embedding_dict, self.feature_index, varlen_sparse_feature_columns, return_dict=True)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, varlen_sparse_feature_columns)
        
        # 连续特征直接保留
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, feature_names=[('sparse', 'var_sparse', 'dense')], feature_group=False):
        '''计算输入维度和，Sparse/VarlenSparse的embedding_dim + Dense的dimesion
        '''
        def get_dim(feat):
            if isinstance(feat, DenseFeat):
                return feat.dimension
            elif feature_group:
                return 1
            else:
                return feat.embedding_dim

        feature_col_groups = split_columns(feature_columns, feature_names)
        input_dim = 0
        for feature_col in feature_col_groups:
            if isinstance(feature_col, list):
                for feat in feature_col:
                    input_dim += get_dim(feat)
            else:
                input_dim += get_dim(feature_col)
                    
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        """记录需要正则的参数项
        """
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        """计算正则损失
        """
        total_reg_loss = 0
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
        sparse_feature_columns = split_columns(feature_columns, ['sparse', 'var_sparse'])
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]


class DeepFM(RecBase):
    """DeepFM的实现
    Reference: [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, use_fm=True, dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=1e-4,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, task=task)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, 
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        # 离散变量过embedding，连续变量保留原值
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)  # [btz, 1]

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)  # [btz, feat_cnt, emb_size]
            # FM仅对离散特征进行交叉
            logit += self.fm(fm_input)  # [btz, 1]

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # [btz, sparse_feat_cnt*emb_size+dense_feat_cnt]
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred


class WideDeep(RecBase):
    """WideDeep的实现
    Wide部分是SparseFeat过embedding, VarlenSparseFeat过embedding+pooling, Dense特征直接取用
    Deep部分所有特征打平[btz, sparse_feat_cnt*emb_size+dense_feat_cnt]过DNN
    Reference: [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=1e-4, 
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', **kwargs):
        super(WideDeep, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, task=task)

        if len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, 
                dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        # SparseFeat和VarLenSparseFeat生成Embedding，VarLenSparseFeat要过Pooling, DenseFeat直接从X中取用
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)  # [btz, 1]

        if hasattr(self, 'dnn') and hasattr(self, 'dnn_linear'):
            # 所有特征打平并concat在一起，[btz, sparse_feat_cnt*emb_size+dense_feat_cnt]
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred


class DeepCross(WideDeep):
    """Deep&Cross
    和Wide&Deep相比，是用CrossNet替换了linear_model
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)
    [2] Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, cross_parameterization='vector',
                 dnn_hidden_units=(256, 128), l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_cross=1e-5,
                 l2_reg_dnn=0, init_std=0.0001, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', use_linear=False, **kwargs):
        super(DeepCross, self).__init__(linear_feature_columns, dnn_feature_columns, **get_kw(DeepCross, locals()))

        # 默认应该不使用linear_model
        if not use_linear:
            del self.linear_model

        dnn_linear_in_feature = 0
        if len(dnn_hidden_units) > 0:
            dnn_linear_in_feature += dnn_hidden_units[-1]
        if cross_num > 0:
            dnn_linear_in_feature += self.compute_input_dim(dnn_feature_columns)
        
        if dnn_linear_in_feature > 0:
            self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)
        self.crossnet = CrossNet(in_features=self.compute_input_dim(dnn_feature_columns),
                                 layer_num=cross_num, parameterization=cross_parameterization)
        self.add_regularization_weight(self.crossnet.kernels, l2=l2_reg_cross)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X) if hasattr(self, 'linear_model') else 0 # [btz, 1]

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # [btz, sparse_feat_cnt*emb_size+dense_feat_cnt]

        # CrossNetwork
        stack_out = [self.crossnet(dnn_input)]

        # Deep Network
        if hasattr(self, 'dnn'):
            stack_out.append(self.dnn(dnn_input))
        stack_out = torch.cat(stack_out, dim=-1)

        if hasattr(self, 'dnn_linear'):
            logit += self.dnn_linear(stack_out)

        # Out
        y_pred = self.out(logit)
        return y_pred

class DIN(RecBase):
    """Deep Interest Network实现
    """
    def __init__(self, dnn_feature_columns, item_history_list, dnn_hidden_units=(256, 128),
                 att_hidden_units=(64, 16), att_activation='Dice', att_weight_normalization=False,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=1e-4,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_embedding=l2_reg_embedding, init_std=init_std, task=task)
        del self.linear_model  # 删除不必要的网络结构
        
        self.sparse_feature_columns, self.dense_feature_columns, self.varlen_sparse_feature_columns = split_columns(dnn_feature_columns)
        self.item_history_list = item_history_list

        # 把varlen_sparse_feature_columns分解成hist、neg_hist和varlen特征
        # 其实是DIEN的逻辑（为了避免多次执行），DIN中少了neg模块，DIEN是在deepctr是在forward中会重复执行多次
        self.history_feature_names = list(map(lambda x: "hist_"+x, item_history_list))
        self.neg_history_feature_names = list(map(lambda x: "neg_" + x, self.history_feature_names))
        self.history_feature_columns = []
        self.neg_history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_feature_names:
                self.history_feature_columns.append(fc)
            elif feature_name in self.neg_history_feature_names:
                self.neg_history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        # Attn模块
        att_emb_dim = self._compute_interest_dim()
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_units, embedding_dim=att_emb_dim, att_activation=att_activation,
                                                       return_score=False, supports_masking=False, weight_normalization=att_weight_normalization)

        # DNN模块
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, 
                       dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)


    def forward(self, X):
        # 过embedding
        emb_lists, query_emb, keys_emb, keys_length, deep_input_emb = self._get_emb(X)

        # 获取变长稀疏特征pooling的结果， [[btz, 1, emb_size]
        sequence_embed_dict = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_varlen_feature_columns)
        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, self.sparse_varlen_feature_columns)
        
        # Attn部分
        hist = self.attention(query_emb, keys_emb, keys_length)  # [btz, 1, hdsz]

        # dnn部分
        dnn_input_emb_list = emb_lists[2]
        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat([deep_input_emb, hist], dim=-1)  # [btz, 1, hdsz]
        dnn_input = combined_dnn_input([deep_input_emb], emb_lists[-1])  # [btz, hdsz]
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        # 输出
        y_pred = self.out(dnn_logit)

        return y_pred
        
    def _get_emb(self, X):
        # 过embedding，这里改造embedding_lookup使得只经过一次embedding, 加快训练速度
        # query_emb_list     [[btz, 1, emb_size], ...]
        # keys_emb_list      [[btz, seq_len, emb_size], ...]
        # dnn_input_emb_list [[btz, 1, emb_size], ...]
        return_feat_list = [self.item_history_list, self.history_feature_names, [fc.name for fc in self.sparse_feature_columns]]
        emb_lists = embedding_lookup(X, self.embedding_dict, self.feature_index, self.dnn_feature_columns, return_feat_list=return_feat_list)
        query_emb_list, keys_emb_list, dnn_input_emb_list = emb_lists
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in self.dense_feature_columns]
        emb_lists.append(dense_value_list)

        query_emb = torch.cat(query_emb_list, dim=-1)  # [btz, 1, hdsz]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [btz, 1, hdsz]
        keys_length = maxlen_lookup(X, self.feature_index, self.history_feature_names)  # [btz, 1]
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)  # [btz, 1, hdsz]
        return emb_lists, query_emb, keys_emb, keys_length, deep_input_emb

    def _compute_interest_dim(self):
        """计算兴趣网络特征维度和
        """
        dim_list = [feat.embedding_dim for feat in self.sparse_feature_columns if feat.name in self.item_history_list]
        return sum(dim_list)


class DIEN(DIN):
    """Deep Interest Evolution Network
    """
    def __init__(self, dnn_feature_columns, item_history_list, gru_type="GRU", use_negsampling=False, alpha=1.0, 
                 dnn_use_bn=False, dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_units=(64, 16), att_activation="relu", 
                 att_weight_normalization=True, l2_reg_embedding=1e-6, l2_reg_dnn=0, dnn_dropout=0, init_std=0.0001, task='binary'):
        super(DIEN, self).__init__(dnn_feature_columns, item_history_list, dnn_hidden_units, att_hidden_units, att_activation, att_weight_normalization, 
                                   l2_reg_embedding, l2_reg_dnn, init_std, dnn_dropout, dnn_activation, dnn_use_bn, task)
        del self.attention
        self.alpha = alpha

        # 兴趣提取层
        input_size = self._compute_interest_dim()
        self.interest_extractor = InterestExtractor(input_size=input_size, use_neg=use_negsampling, init_std=init_std)

        # 兴趣演变层
        self.interest_evolution = InterestEvolving(input_size=input_size, gru_type=gru_type, use_neg=use_negsampling, init_std=init_std,
                                                   att_hidden_size=att_hidden_units, att_activation=att_activation, att_weight_normalization=att_weight_normalization)
        
        # DNN
        dnn_input_size = self.compute_input_dim(dnn_feature_columns, [('sparse', 'dense')]) + input_size
        self.dnn = DNN(dnn_input_size, dnn_hidden_units, activation=dnn_activation, 
                       dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std)

    def forward(self, X):
        # 过embedding
        emb_lists, query_emb, keys_emb, keys_length, deep_input_emb = self._get_emb(X)
        neg_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.dnn_feature_columns, return_feat_list=self.neg_history_feature_names)
        neg_keys_emb = torch.cat(neg_keys_emb_list, dim=-1)  # [btz, 1, hdsz]

        # 过兴趣提取层
        # input shape: [btz, seq_len, hdsz],  [btz, 1], [btz, seq_len, hdsz]
        # masked_interest shape: [btz, seq_len, hdsz]
        masked_interest, aux_loss = self.interest_extractor(keys_emb, keys_length, neg_keys_emb)
        self.add_auxiliary_loss(aux_loss, self.alpha)

        # 过兴趣演变层
        hist = self.interest_evolution(query_emb, masked_interest, keys_length)  # [btz, hdsz]

        # dnn部分
        deep_input_emb = torch.cat([deep_input_emb.squeeze(1), hist], dim=-1)  # [btz, hdsz]
        dnn_input = combined_dnn_input([deep_input_emb], emb_lists[-1])  # [btz, hdsz]
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        # 输出
        y_pred = self.out(dnn_logit)

        return y_pred