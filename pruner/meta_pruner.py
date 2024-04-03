import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil
from collections import OrderedDict
from utils import strdict_to_dict

class Layer:
    def __init__(self, name, size, layer_index, res=False):
        self.name = name
        self.size = []
        for x in size:
            self.size.append(x)
        self.layer_index = layer_index
        self.is_shortcut = True if "downsample" in name else False
        if res:
            self.stage, self.seq_index, self.block_index = self._get_various_index_by_name(name)
    
    def _get_various_index_by_name(self, name):
        '''Get the indeces including stage, seq_ix, blk_ix.
            Same stage means the same feature map size.
        '''
        global lastest_stage # an awkward impel, just for now
        if name.startswith('module.'):
            name = name[7:] # remove the prefix caused by pytorch data parallel

        if "conv1" == name: # TODO: this might not be so safe
            lastest_stage = 0
            return 0, None, None
        if "linear" in name or 'fc' in name: # Note: this can be risky. Check it fully. TODO: @mingsun-tse
            return lastest_stage + 1, None, None # fc layer should always be the last layer
        else:
            try:
                stage  = int(name.split(".")[0][-1]) # ONLY work for standard resnets. name example: layer2.2.conv1, layer4.0.downsample.0
                seq_ix = int(name.split(".")[1])
                if 'conv' in name.split(".")[-1]:
                    blk_ix = int(name[-1]) - 1
                else:
                    blk_ix = -1 # shortcut layer  
                lastest_stage = stage
                return stage, seq_ix, blk_ix
            except:
                print('!Parsing the layer name failed: %s. Please check.' % name)
                
class MetaPruner:
    def __init__(self, model, args, logger, passer):
        self.model = model
        self.args = args
        self.logger = logger
        self.logprint = logger.log_printer.logprint
        self.accprint = logger.log_printer.accprint
        self.netprint = logger.log_printer.netprint
        self.test = lambda net: passer.test(passer.test_loader, net, passer.criterion, passer.args)
        self.train_loader = passer.train_loader
        self.criterion = passer.criterion
        self.save = passer.save
        self.is_single_branch = passer.is_single_branch
        
        self.layers = OrderedDict()
        self._register_layers()

        arch = self.args.arch
        if arch.startswith('resnet'):
            # TODO: add block
            self.n_conv_within_block = 0
            if args.dataset == "imagenet":
                if arch in ['resnet18', 'resnet34']:
                    self.n_conv_within_block = 2
                elif arch in ['resnet50', 'resnet101', 'resnet152']:
                    self.n_conv_within_block = 3
            else:
                self.n_conv_within_block = 2

        self.kept_wg = {}
        self.pruned_wg = {}
        self.get_pr() # set up pr for each layer
        
    def _pick_pruned(self, w_abs, pr, mode="min"):
        if pr == 0:
            return []
        w_abs_list = w_abs.flatten()
        n_wg = len(w_abs_list)
        n_pruned = min(ceil(pr * n_wg), n_wg - 1) # do not prune all
        if mode == "rand":
            out = np.random.permutation(n_wg)[:n_pruned]
        elif mode == "min":
            out = w_abs_list.sort()[1][:n_pruned]
        elif mode == "max":
            out = w_abs_list.sort()[1][-n_pruned:]
        return out

    def _register_layers(self):
        '''
            This will maintain a data structure that can return some useful 
            information by the name of a layer.
        '''
        ix = -1 # layer index, starts from 0
        max_len_name = 0
        layer_shape = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if "downsample" not in name:
                    ix += 1
                layer_shape[name] = [ix, m.weight.size()]
                max_len_name = max(max_len_name, len(name))
                
                size = m.weight.size()
                res = True if self.args.arch.startswith('resnet') else False
                self.layers[name] = Layer(name, size, ix, res)
        
        max_len_ix = len("%s" % ix)
        print("Register layer index and kernel shape:")
        format_str = "[%{}d] %{}s -- kernel_shape: %s".format(max_len_ix, max_len_name)
        for name, (ix, ks) in layer_shape.items():
            print(format_str % (ix, name, ks))

    def _next_conv(self, model, name, mm):
        if hasattr(self.layers[name], 'block_index'):
            block_index = self.layers[name].block_index
            if block_index == self.n_conv_within_block - 1:
                return None
        ix_conv = 0
        ix_mm = -1
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                ix_conv += 1
                if m == mm:
                    ix_mm = ix_conv
                if ix_mm != -1 and ix_conv == ix_mm + 1:
                    return n
        return None
    
    def _prev_conv(self, model, name, mm):
        if hasattr(self.layers[name], 'block_index'):
            block_index = self.layers[name].block_index
            if block_index in [None, 0, -1]: # 1st conv, 1st conv in a block, 1x1 shortcut layer
                return None
        for n, _ in model.named_modules():
            if n in self.layers:
                ix = self.layers[n].layer_index
                if ix + 1 == self.layers[name].layer_index:
                    return n
        return None

    def _next_bn(self, model, mm):
        just_passed_mm = False
        for m in model.modules():
            if m == mm:
                just_passed_mm = True
            if just_passed_mm and isinstance(m, nn.BatchNorm2d):
                return m
        return None
   
    def _replace_module(self, model, name, new_m):
        '''
            Replace the module <name> in <model> with <new_m>
            E.g., 'module.layer1.0.conv1'
            ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
        '''
        obj = model
        segs = name.split(".")
        for ix in range(len(segs)):
            s = segs[ix]
            if ix == len(segs) - 1: # the last one
                if s.isdigit():
                    obj.__setitem__(int(s), new_m)
                else:
                    obj.__setattr__(s, new_m)
                return
            if s.isdigit():
                obj = obj.__getitem__(int(s))
            else:
                obj = obj.__getattr__(s)
    
    def _get_n_filter(self, model):
        '''
            Do not consider the downsample 1x1 shortcuts.
        '''
        n_filter = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                if not self.layers[name].is_shortcut:
                    n_filter.append(m.weight.size(0))
        return n_filter
    
    def _get_layer_pr_vgg(self, name):
        '''Example: '[0-4:0.5, 5:0.6, 8-10:0.2]'
                    6, 7 not mentioned, default value is 0
        '''
        layer_index = self.layers[name].layer_index
        pr = self.args.stage_pr[layer_index]
        if str(layer_index) in self.args.skip_layers:
            pr = 0
        return pr

    def _get_layer_pr_resnet(self, name):
        '''
            This function will determine the prune_ratio (pr) for each specific layer
            by a set of rules.
        '''
        wg = self.args.wg
        layer_index = self.layers[name].layer_index
        stage = self.layers[name].stage
        seq_index = self.layers[name].seq_index
        block_index = self.layers[name].block_index
        is_shortcut = self.layers[name].is_shortcut
        pr = self.args.stage_pr[stage]

        # for unstructured pruning, no restrictions, every layer can be pruned
        if self.args.wg != 'weight':
            # do not prune the shortcut layers for now
            if is_shortcut:
                pr = 0
            
            # do not prune layers we set to be skipped
            layer_id = '%s.%s.%s' % (str(stage), str(seq_index), str(block_index))
            for s in self.args.skip_layers:
                if s and layer_id.startswith(s):
                    pr = 0

            # for channel/filter prune, do not prune the 1st/last conv in a block
            if (wg == "channel" and block_index == 0) or \
                (wg == "filter" and block_index == self.n_conv_within_block - 1):
                pr = 0
        
        # Deprecated, will be removed:
        # # adjust accordingly if we explictly provide the pr_ratio_file
        # if self.args.pr_ratio_file:
        #     line = open(self.args.pr_ratio_file).readline()
        #     pr_weight = strdict_to_dict(line, float)
        #     if str(layer_index) in pr_weight:
        #         pr = pr_weight[str(layer_index)] * pr
        return pr
        
    def get_pr(self):
        # 判断所使用的架构是否为单分支架构（如VGG），如果是，则使用针对VGG的剪枝方法。
        if self.is_single_branch(self.args.arch):
            get_layer_pr = self._get_layer_pr_vgg
        # 否则，使用针对ResNet的剪枝方法。
        else:
            get_layer_pr = self._get_layer_pr_resnet

        # 初始化存储剪枝比例的字典。
        self.pr = {}

        # 如果设置了阶段性剪枝参数，表示当前是剪枝过程中的一个阶段。
        if self.args.stage_pr:  # 如果提供了基础剪枝模型，则stage_pr可能为None。
            # 遍历模型的所有模块，并为卷积层和全连接层计算剪枝比例。
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # 调用对应的剪枝方法来获取剪枝比例，并存储在字典中。
                    self.pr[name] = get_layer_pr(name)
        # 如果没有设置阶段性剪枝参数，则需要加载提供的基础剪枝模型。
        else:
            # 断言确保提供了基础剪枝模型的路径。
            assert self.args.base_pr_model
            # 加载基础剪枝模型的状态。
            state = torch.load(self.args.base_pr_model)
            # 获取并存储已剪枝和未剪枝权重组的信息。
            self.pruned_wg_pr_model = state['pruned_wg']
            self.kept_wg_pr_model = state['kept_wg']
            # 计算并存储每层的剪枝比例。
            for k in self.pruned_wg_pr_model:
                n_pruned = len(self.pruned_wg_pr_model[k])
                n_kept = len(self.kept_wg_pr_model[k])
                # 剪枝比例计算为已剪枝权重数量除以总权重数量。
                self.pr[k] = float(n_pruned) / (n_pruned + n_kept)
            # 输出日志信息，提示基础剪枝模型加载成功，并继承了其剪枝比例。
            self.logprint("==> Load base_pr_model successfully and inherit its pruning ratio: '{}'".format(self.args.base_pr_model))

    def _get_kept_wg_L1(self):
        # 检查是否提供了基础剪枝模型，且继承方式为'index'。
        if self.args.base_pr_model and self.args.inherit_pruned == 'index':
            # 如果是，直接继承基础剪枝模型中的剪枝和保留权重组。
            self.pruned_wg = self.pruned_wg_pr_model
            self.kept_wg = self.kept_wg_pr_model
            # 记录日志信息，指出从基础剪枝模型继承了剪枝索引。
            self.logprint("==> Inherit the pruned index from base_pr_model: '{}'".format(self.args.base_pr_model))
        else:
            # 如果不继承或没有提供基础剪枝模型，根据权重组(wg)参数决定剪枝策略。
            wg = self.args.wg
            # 遍历模型中的所有模块。
            for name, m in self.model.named_modules():
                # 只对卷积层和全连接层进行操作。
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # 获取权重的形状。
                    shape = m.weight.data.shape
                    # 根据权重组参数计算每个权重的分数（基于L1范数）。
                    if wg == "filter":
                        score = m.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=1)
                    elif wg == "channel":
                        score = m.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=0)
                    elif wg == "weight":
                        score = m.weight.abs().flatten()
                    else:
                        # 如果wg参数不是这三种之一，抛出未实现错误。
                        raise NotImplementedError
                    # 根据分数和剪枝比例确定哪些权重将被剪枝，并将其索引存储在self.pruned_wg中。
                    self.pruned_wg[name] = self._pick_pruned(score, self.pr[name], self.args.pick_pruned)
                    # 计算并存储应该被保留的权重的索引。
                    self.kept_wg[name] = [i for i in range(len(score)) if i not in self.pruned_wg[name]]
                    # 记录关于该层剪枝情况的日志信息。
                    logtmp = '[%2d %s] got pruned wg by L1 sorting (%s), pr %.4f' % (self.layers[name].layer_index, name, self.args.pick_pruned, self.pr[name])
                    
                    # 如果提供了基础剪枝模型，比较基于L1排序选出的剪枝权重与基础模型中的剪枝权重的交集比例。
                    if self.args.base_pr_model:
                        intersection = [x for x in self.pruned_wg_pr_model[name] if x in self.pruned_wg[name]]
                        intersection_ratio = len(intersection) / len(self.pruned_wg[name]) if len(self.pruned_wg[name]) else 0
                        logtmp += ', intersection ratio of the weights picked by L1 vs. base_pr_model: %.4f (%d)' % (intersection_ratio, len(intersection))
                    # 输出剪枝情况的详细日志。
                    self.netprint(logtmp)

    def _get_kept_filter_channel(self, m, name):
        # 根据剪枝的策略是基于通道还是过滤器来决定保留的通道和过滤器
        if self.args.wg == "channel":
            # 如果是基于通道的剪枝策略
            kept_chl = self.kept_wg[name]  # 获取当前卷积层保留的通道索引
            next_conv = self._next_conv(self.model, name, m)  # 找到当前卷积层的下一个卷积层
            if not next_conv:
                # 如果没有下一个卷积层（即当前层是最后一个卷积层）
                kept_filter = range(m.weight.size(0))  # 所有的过滤器都被保留
            else:
                # 如果有下一个卷积层
                kept_filter = self.kept_wg[next_conv]  # 保留的过滤器索引与下一个卷积层保留的通道索引相同
            
        elif self.args.wg == "filter":
            # 如果是基于过滤器的剪枝策略
            kept_filter = self.kept_wg[name]  # 获取当前卷积层保留的过滤器索引
            prev_conv = self._prev_conv(self.model, name, m)  # 找到当前卷积层的前一个卷积层
            if not prev_conv:
                # 如果没有前一个卷积层（即当前层是第一个卷积层）
                kept_chl = range(m.weight.size(1))  # 所有的通道都被保留
            else:
                # 如果有前一个卷积层
                kept_chl = self.kept_wg[prev_conv]  # 保留的通道索引与前一个卷积层保留的过滤器索引相同
            
        return kept_filter, kept_chl  # 返回保留的过滤器和通道的索引列表



    def _prune_and_build_new_model(self):
        # 如果参数指定了权重组为'weight'，则仅获取掩码并返回
        if self.args.wg == 'weight':
            self._get_masks()
            return

        # 深拷贝当前模型以构建新模型
        new_model = copy.deepcopy(self.model)
        # 用于记录线性层的数量
        cnt_linear = 0
        # 遍历当前模型的所有模块及其名称
        for name, m in self.model.named_modules():
            # 如果模块是卷积层
            if isinstance(m, nn.Conv2d):
                # 获取要保留的卷积核(filter)和通道
                kept_filter, kept_chl = self._get_kept_filter_channel(m, name)
                
                # 复制卷积层的权重和偏置
                bias = False if isinstance(m.bias, type(None)) else True
                kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]
                new_conv = nn.Conv2d(kept_weights.size(1), kept_weights.size(0), m.kernel_size,
                                m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                new_conv.weight.data.copy_(kept_weights)  # 将权重加载到新模块
                if bias:
                    kept_bias = m.bias.data[kept_filter]
                    new_conv.bias.data.copy_(kept_bias)
                
                # 加载新的卷积层
                self._replace_module(new_model, name, new_conv)

                # 获取对应的批量归一化层（如果有的话）以供后用
                next_bn = self._next_bn(self.model, m)

            # 如果模块是批量归一化层，并且是紧接在卷积层之后的归一化层
            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                new_bn = nn.BatchNorm2d(len(kept_filter), eps=m.eps, momentum=m.momentum, 
                        affine=m.affine, track_running_stats=m.track_running_stats).cuda()
                
                # 复制批量归一化层的权重和偏置
                if self.args.copy_bn_w:
                    weight = m.weight.data[kept_filter]
                    new_bn.weight.data.copy_(weight)
                if self.args.copy_bn_b:
                    bias = m.bias.data[kept_filter]
                    new_bn.bias.data.copy_(bias)
                
                # 复制批量归一化层的运行时统计数据
                new_bn.running_mean.data.copy_(m.running_mean[kept_filter])
                new_bn.running_var.data.copy_(m.running_var[kept_filter])
                new_bn.num_batches_tracked.data.copy_(m.num_batches_tracked)
                
                # 加载新的批量归一化层
                self._replace_module(new_model, name, new_bn)
            
            # 如果模块是全连接层
            elif isinstance(m, nn.Linear):
                cnt_linear += 1
                # 如果是第一个全连接层
                if cnt_linear == 1:
                    # 获取最后一个卷积层
                    last_conv = ''
                    last_conv_name = ''
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, nn.Conv2d):
                            last_conv = mm
                            last_conv_name = n
                    kept_filter_last_conv, _ = self._get_kept_filter_channel(last_conv, last_conv_name)
                    
                    # 获取要保持的权重
                    dim_in = m.weight.size(1)
                    fm_size = int(dim_in / last_conv.weight.size(0))  # 例如对于alexnet是36
                    kept_dim_in = []
                    for i in kept_filter_last_conv:
                        tmp = list(range(i * fm_size, i * fm_size + fm_size))
                        kept_dim_in += tmp
                    kept_weights = m.weight.data[:, kept_dim_in]
                    
                    # 构建新的全连接层
                    bias = False if isinstance(m.bias, type(None)) else True
                    new_linear = nn.Linear(in_features=len(kept_dim_in), out_features=m.out_features, bias=bias).cuda()
                    new_linear.weight.data.copy_(kept_weights)
                    if bias:
                        new_linear.bias.data.copy_(m.bias.data)
                    
                    # 加载新的全连接层
                    self._replace_module(new_model, name, new_linear)
        
        # 更新模型为剪枝后的新模型
        self.model = new_model
        # 获取新模型中的过滤器数量
        n_filter = self._get_n_filter(self.model)
        # 打印过滤器数量
        self.logprint(n_filter)

    def _get_masks(self):
        '''Get masks for unstructured pruning
        '''
        self.mask = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mask = torch.ones_like(m.weight.data).cuda().flatten()
                pruned = self.pruned_wg[name]
                mask[pruned] = 0
                self.mask[name] = mask.view_as(m.weight.data)
        self.logprint('Get masks done for weight pruning')