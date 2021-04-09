import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
VCR_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'data/vcr1', 'vcr1images')
VCR_ANNOTS_DIR = os.path.join(os.path.dirname(__file__), '/data/scene_understanding/R2c')#dataloaders')
VCR_ANNOTS_DIR_json = os.path.join(os.path.dirname(__file__), 'dataloaders')
# Answer_Train_Feature_DIR = "/home/tangxueq/tmp/my_dataset/answer/train"


if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")

