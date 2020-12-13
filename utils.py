import math
import shutil
import os.path as osp
import torch
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

# DALI data loader
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, 0, seed=12)
        #self.input = ops.FileReader(
        #                file_root=osp.join(data_dir, 'train'),
        #                random_shuffle=True)
        self.input = ops.MXNetReader(
                        path=osp.join(data_dir, 'train.rec'),
                        index_path=osp.join(data_dir, 'train.idx'),
                        random_shuffle=True)
        # let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.ImageDecoderRandomCrop(device=dali_device, output_type=types.RGB)
            self.res = ops.Resize(resize_x=crop, resize_y=crop)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle
            # all images from full-sized ImageNet without additional reallocations
            self.decode = ops.ImageDecoder(device="mixed",
                                           output_type=types.RGB,
                                           device_memory_padding=211025920,
                                           host_memory_padding=140544512)
            self.res = ops.RandomResizedCrop(device=dali_device, size =(crop, crop))

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, 0, seed=12)
        #self.input = ops.FileReader(
        #                file_root=osp.join(data_dir, 'val'),
        #                random_shuffle=False)
        self.input = ops.MXNetReader(
                        path=osp.join(data_dir, 'val.rec'),
                        index_path=osp.join(data_dir, 'val.idx'),
                        random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

'''
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, 0, seed=12)
        self.input = ops.MXNetReader(path=osp.join(data_dir, 'train.rec'),
                                     index_path=osp.join(data_dir, 'train.idx'),
                                     random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle
        # all images from full-sized ImageNet without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, 0, seed=12)
        self.input = ops.MXNetReader(path=osp.join(data_dir, 'val.rec'),
                                     index_path=osp.join(data_dir, 'val.idx'),
                                     random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
'''

def get_data_loader(args):
    assert osp.isdir(args.data), '{} does not exist'.format(args.data)

    crop_size = 224
    val_size = 256

    # train loader
    pipe = HybridTrainPipe(batch_size=args.batch_size,
                           num_threads=args.workers,
                           data_dir=args.data,
                           crop=crop_size,
                           dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
    # val loader
    pipe = HybridValPipe(batch_size=args.batch_size,
                         num_threads=args.workers,
                         data_dir=args.data,
                         crop=crop_size,
                         size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))

    return train_loader, val_loader

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

''''
def adjust_learning_rate(args, optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
'''

def adjust_learning_rate(args, optimizer, epoch, step, len_epoch):
    cur_iters = epoch * len_epoch + step + 1
    total_iters = (args.epochs - args.warmup_epochs) * len_epoch
    # warm up lr schedule
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        lr = args.lr * 1.0 * cur_iters / (args.warmup_epochs * len_epoch)
    elif args.lr_mode == 'cos':
        cur_iters = cur_iters - args.warmup_epochs * len_epoch - 1
        lr = 0.5 * args.lr * (1 + math.cos(1.0 * cur_iters / total_iters * math.pi))
    elif args.lr_mode == 'poly':
        cur_iters = cur_iters - args.warmup_epochs * len_epoch - 1
        lr = args.lr * pow((1 - 1.0 * cur_iters / total_iters), 0.9)
    elif args.lr_mode == 'step':
        factor = 0
        for m in eval(args.milestones):
            if epoch >= m:
                factor = factor + 1
        lr = args.lr * (0.1 ** factor)
    else:
        raise NotImplemented
    assert lr >= 0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = output.detach()
    target = target.detach()
    output.requires_grad = False
    target.requires_grad = False

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
