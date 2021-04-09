"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""

import numpy as np
import json
import os
from config import VCR_ANNOTS_DIR,VCR_ANNOTS_DIR_json
import argparse
import pickle
parser = argparse.ArgumentParser(description='Evaluate question -> answer and rationale')
parser.add_argument(
    '-answer_preds',
    dest='answer_preds',
    # default='saves/GA3a/valpreds.npy',
    # default='/home/tangxueq/tmp/lstmdbase_a/valyy.npy',
    # default='/home/tangxueq/tmp/baseline/valpreds.npy',
    default='/phys/ssd/tangxueq/tmp/64qacatnomask/valpreds.npy',
    help='Location of question->answer predictions',
    type=str,
)
parser.add_argument(
    '-rationale_preds',
    dest='rationale_preds',
    # default='saves/GA3r/valpreds.npy',
    # default='/phys/ssd/tangxueq/tmp/64r/valpreds.npy',
    default='/phys/ssd/tangxueq/tmp/rlstmd/valpreds.npy',
    # default='/home/tangxueq/tmp/baseline_rowan_r/valpreds.npy',
    # default='/home/tangxueq/MA_tang/r2c/models/saves/test1/valpreds.npy',
    help='Location of question+answer->rationale predictions',
    type=str,
)
parser.add_argument(
    '-split',
    dest='split',
    default='val',
    help='Split you\'re using. Probably you want val.',
    type=str,
)

args = parser.parse_args()

answer_preds = np.load(args.answer_preds)
rationale_preds = np.load(args.rationale_preds)

rationale_labels = []
answer_labels = []

with open(os.path.join(VCR_ANNOTS_DIR_json, '{}.jsonl'.format(args.split)), 'r') as f:
    for l in f:
        item = json.loads(l)
        answer_labels.append(item['answer_label'])
        rationale_labels.append(item['rationale_label'])
###############################################################################################
# # root = '/phys/ssd/tangxueq/tmp/data/vcr/vcrimage'
# root = '/phys/ssd/tangxueq/tmp/data/VCRdataset'
# # root = '/phys/ssd/tangxueq/tmp/data/VCR'
# # root = '/phys/ssd/tangxueq/tmp/data/VCRDataset'
# # root = '/localstorage/tangxueq/VCR'
# # root = '/phys/ssd/tangxueq/tmp/data/VCRdataset'
#
#
# def load_filenames(root):
#     files = [x for x in os.listdir(root) if x.split('.')[-1] == 'npy']
#     files = sorted(files, key=lambda x: str(x.split('.')[-2][0:]), reverse=False)
#     files = [os.path.join(root, x) for x in files]
#     return files

# answer_items_path = load_filenames(os.path.join(root, 'answer', 'val'))
# rationale_items_path = load_filenames(os.path.join(root, 'rationale', 'val'))
# def loader(items_path,labels):
#
#     for p in items_path:
#         with open(p, "rb") as file:
#             data_dict = pickle.load(file)
#
#             labels.append(data_dict['detector']['label'])
#         # raise InterruptedError
#
#     return labels
# answer_labels = loader(answer_items_path,answer_labels)
# rationale_labels = loader(rationale_items_path,rationale_labels)

###################################################################################################
answer_labels = np.array(answer_labels)
rationale_labels = np.array(rationale_labels)



# Sanity checks
assert answer_preds.shape[0] == answer_labels.size
assert rationale_preds.shape[0] == answer_labels.size
assert answer_preds.shape[1] == 4
assert rationale_preds.shape[1] == 4

answer_hits = answer_preds.argmax(1) == answer_labels
print(answer_hits)
rationale_hits = rationale_preds.argmax(1) == rationale_labels
joint_hits = answer_hits & rationale_hits

print("Answer acc:    {:.3f}".format(np.mean(answer_hits)), flush=True)
print("Rationale acc: {:.3f}".format(np.mean(rationale_hits)), flush=True)
print("Joint acc:     {:.3f}".format(np.mean(answer_hits & rationale_hits)), flush=True)