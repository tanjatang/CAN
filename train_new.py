"""
Training script. Should be pretty adaptable to whatever.
"""
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../,,/")
sys.path.append("../../../,,/../")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import shutil
# import ipdb


import torch.optim as optim
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from dataloaders.vcr_replace_image import VCR, VCRLoader
from utils.pytorch_misc_dict import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models
# import opts
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

from tensorboard_logger import Logger

logger = Logger(logdir="./tensorboard_logs", flush_secs=10)
#################################
#################################
######## Data loading stuff
#################################
#################################
'''
parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    default='models/multiatt/default.json',
    type=str,
)
parser.add_argument(
    '-rationale',
    default=True,
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    default='models/saves/flagship_rationale',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)



'''
parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

#
# parser.add_argument(
#     '--memory_cell_path', dest='memory_cell_path', type=str, default='0',
#                         help='memory cell path')
#

# opt = opts.parse_opt()
args = parser.parse_args()
# opt = args.memory_cell_path
print(str(args.params))
params = Params.from_file(args.params)
# ipdb.set_trace()
train, val, test = VCR.splits(mode='rationale' if args.rationale else 'answer',
                              embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True))


#q确定模式为train，val还是test，从bert下载相应的数据
NUM_GPUS = 1#torch.cuda.device_count()

NUM_CPUS = 4#multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")


num_workers = 16#(4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1

print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 64// NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers} #96###############################################
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
test_loader = VCRLoader.from_dataset(test, **loader_params)




'''use for debugging'''

################################################################################
# if __name__ == '__main__':
#     # torch.multiprocessing.set_start_method('spawn')
#     print(train_loader,"KKKKKKKKKKKKKKKKKKKKKK")
#     collect = next(iter(train_loader))
#
#     for k, v in collect.items():
#         print(k, v)
#
#     # collect = next(iter(train_loader))
#     # print(len(collect),next(iter(collect)))
#     # # for k,v in collect.items():
#     # #     print(k, v)
#
#     raise ValueError("........................................")
##################################################################################

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:

        if k != 'metadata':

            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True) #transfer tensor to cuda
    return td



ARGS_RESET_EVERY = 100
print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)





model = Model.from_params(vocab=train.vocab, params=params['model'])



for submodule in model.detector.backbone.modules():
    if isinstance(submodule, BatchNorm2d):#每个通道上归一化特征
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False#这层的参数不更新


model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()




optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad and x[0] != 'memory_cell'] ,
                                  params['trainer']['optimizer'])#在default里面，用Adam优化，


optimizer_mem = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad and x[0] == 'memory_cell'] ,
                                  params['trainer']['optimizer_mem'])#在default里面，用Adam优化，

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None#


if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, optimizer_mem, serialization_dir=args.folder,
                                                        learning_rate_scheduler=scheduler)#？

else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)

# opti_path='/home/tangxueq/MA_tang/r2c/models/saves'
# if os.path.isfile(os.path.join(opti_path, 'optimizer_mem' + format(9) + '.pth')):
#     print("optimizer memory!!!!!!!!!!!!!!")
#     # optimizer_mem.load_state_dict(torch.load(opti_path))
#     optimizer_mem.load_state_dict(torch.load(os.path.join(
#         opti_path, 'optimizer_mem' + '.pth')))

param_shapes = print_para(model)
num_batches = 0
# opt = opts.parse_opt()###############################################


val_loss = []
train_acc = []
train_loss = []
# restore_best_checkpoint(model, args.folder)

from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
    #
    # if epoch_num < 10:
    #     training_mode = 0
    #
    # else:
    #     training_mode = 1
    # training_mode = 1
# for epoch_num in range(start_epoch, 1 + start_epoch):

    ###########################################################

    # memory_cell_path = '/home/tangxueq/MA_tang/r2c/models/saves/memory_cell.npz'
    #
    # if os.path.isfile(memory_cell_path):
    #     print('load memory_cell from {0}'.format(memory_cell_path))
    #     memory_init = np.load(memory_cell_path)['memory_cell']
    #     # memory_init = np.load(self.memory_cell_path)['memory_cell'][()]
    #
    #     print(type(memory_init),"hhhhhhhhhhhhhhhhhhhhhhhhhhh")
    # else:
    #     print('create a new memory_cell')
    #     memory_init = np.random.rand(5000, 1024) / 100
    #
    # memory_init = np.float32(memory_init)
    #
    #
    # memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()
    # print(memory_cell)



    ##########################################################


    train_results = []
    norms = []
    model.train()


    for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        ################

        batch = _to_gpu(batch)


        optimizer.zero_grad()
        optimizer_mem.zero_grad()



        output_dict = model(**batch)

        loss = output_dict['loss'].mean() + output_dict['cnn_regularization_loss'].mean()
        loss.backward()

        num_batches += 1
        if scheduler:
            scheduler.step_batch(num_batches)#

        norms.append(
            clip_grad_norm(model.named_parameters(), max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
        )
        optimizer.step()
        optimizer_mem.step()

        train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
                                        'crl': output_dict['cnn_regularization_loss'].mean().item(),
                                        'accuracy': (model.module if NUM_GPUS > 1 else model).get_metrics(
                                            reset=(b % ARGS_RESET_EVERY) == 0)[
                                            'accuracy'],
                                        'sec_per_batch': time_per_batch,
                                        'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                        }))



        if b % ARGS_RESET_EVERY == 0 and b > 0:
            norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

            print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ), flush=True)


            #
            # logger.log_value('train acc', (model.module if NUM_GPUS > 1 else model).get_metrics(
            #     reset=(b % ARGS_RESET_EVERY) == 0)[
            #     'accuracy'], epoch_num * len(train_loader) + b)
            # logger.log_value('train loss', output_dict['loss'].mean().item(), epoch_num * len(train_loader) + b)




    print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))
    train_acc.append((model.module if NUM_GPUS > 1 else model).get_metrics(
        reset=(b % ARGS_RESET_EVERY) == 0)[
                         'accuracy'])

    train_loss.append(output_dict['loss'].mean().item())
    val_probs = []
    val_labels = []
    val_loss_sum = 0.0
    model.eval()
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)

            output_dict = model(**batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            # print("jjjj", val_probs, 'kkkkk', batch['label'])
            val_labels.append(batch['label'].detach().cpu().numpy())
            val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]


        #############################################

        print('accuracy:' ,(model.module if NUM_GPUS > 1 else model).get_metrics(
        reset = (b % ARGS_RESET_EVERY) == 0)[
        'accuracy'])
        # #
        # # print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        # logger.log_value('val acc', (model.module if NUM_GPUS > 1 else model).get_metrics(
        # reset = (b % ARGS_RESET_EVERY) == 0)[
        # 'accuracy'], epoch_num * len(val_loader) + b)
        # logger.log_value('val loss',output_dict['loss'].mean().item(), epoch_num * len(val_loader) + b)


    ########################################################

    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    # print("jjjj",val_labels,'kkkkk',val_probs)
    val_loss_avg = val_loss_sum / val_labels.shape[0]

    val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
    if scheduler:
        scheduler.step(val_metric_per_epoch[-1], epoch_num)

    val_loss.append(val_loss_avg)
    print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
          flush=True)
    if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
        print("Stopping at epoch {:2d}".format(epoch_num))
        break######################################################################################

    save_checkpoint(model, optimizer, optimizer_mem,args.folder, epoch_num, val_metric_per_epoch,
                    is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))



    ########################
    path = '/home/tangxueq/tmp/memory_cell800.npz'
    memory_cell = model.memory_cell.data.cpu().numpy()
    print(memory_cell)

    np.savez(path, memory_cell=memory_cell)




y1 = val_metric_per_epoch
y11 = train_acc
y2 = val_loss
y22 = train_loss
    # (model.module if NUM_GPUS > 1 else model).get_metrics(
    #                                         reset=(b % ARGS_RESET_EVERY) == 0)[
    #                                         'accuracy']
# x1 = range(0, 200)
# x2 = range(0, 200)
# y1 = Accuracy_list
# y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot( y1, 'o-')
plt.plot(y11,'x-')

plt.title('accuracy vs. epoches')
plt.ylabel(' accuracy')
plt.legend(["val",'train'],loc ='upper right')
plt.subplot(2, 1, 2)
plt.plot( y2, 'o-')
plt.plot(y22,'x-')

plt.xlabel('loss vs. epoches')
plt.ylabel('loss')
plt.legend(["val",'train'],loc ='upper right')
plt.show()
plt.savefig("tr+d.png")

#
# pyplot.plot((model.module if NUM_GPUS > 1 else model).get_metrics(
#                 reset=(b % ARGS_RESET_EVERY) == 0)[
#                                  'accuracy'])
# pyplot.plot(val_metric_per_epoch)
#
# pyplot.title('model train vs validation loss')
# pyplot.ylabel('loss')
# pyplot.xlabel('epoch')
# pyplot.legend(['train', 'validation'], loc='upper right')
# pyplot.show()
# pyplot.savefig('image.png')
# ####################################################

print("STOPPING. now running the best model on the validation set", flush=True)
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
val_probs = []
val_labels = []
for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    with torch.no_grad():
        batch = _to_gpu(batch)
        output_dict = model(**batch)
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch['label'].detach().cpu().numpy())
val_labels = np.concatenate(val_labels, 0)
val_probs = np.concatenate(val_probs, 0)
acc = float(np.mean(val_labels == val_probs.argmax(1)))
print("Final val accuracy is {:.3f}".format(acc))
np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)





