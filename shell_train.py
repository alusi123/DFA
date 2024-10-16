import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--target", "-t", default="cartoon", help="Target")
parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")

args = parser.parse_args()

###############################################################################

source = ["photo", "cartoon", "art_painting", "sketch"]
input_dir = "PACS/datalists"
image_dir = "PACS/kfold/"

##############################################################################
target = args.target
source.remove(target)
os.system(
            f'CUDA_VISIBLE_DEVICES={args.gpu} '
            f'python train.py '
            f'--source {source[0]} {source[1]} {source[2]} '
            f'--target {target} '
            f'--exp_name pacs '
            f'--seed '
            f'--input_dir {input_dir} '
            f'--image_dir {image_dir} '
            )
    


