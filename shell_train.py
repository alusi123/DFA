import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--target", "-t", default="cartoon", help="Target")
parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")

args = parser.parse_args()

###############################################################################

source = ["photo", "cartoon", "art_painting", "sketch"]

##############################################################################
target = args.domain
source.remove(target)
for i in range(args.times):
    os.system(
              f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python train_newpacs.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--exp_name pacs '
              f'--cl_cd '
              f'--cl_cr '
              f'--cl_weight 0.001 '
              f'--inv_weight 1 '
              f'--seed '
              )
    


