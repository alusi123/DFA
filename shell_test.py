import os
import wget
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--target", "-t", default="sketch", help="Target")
parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")

args = parser.parse_args()

###############################################################################

url = {
    'art_painting': 'https://drive.google.com/uc?export=download&id=1PcStJqu8hUfv4sabAQKULxXM5e111nXX',
    'cartoon'     : 'https://drive.google.com/uc?export=download&id=1x0O6_v9g_u8cS_DiKTtxqfqNMaeuI83n',
    'photo'       : 'https://drive.google.com/uc?export=download&id=134b-SAsCBDW0opw1F-mkkwXxVQuVnmM1',
    'sketch'      : 'https://drive.google.com/uc?export=download&id=16MZrZE1kQBk9JcVGDwRpSBYFi_pl3I2f',
}

source = ["photo", "cartoon", "art_painting", "sketch"]
target = args.target
source.remove(target)

config = "PACS/ResNet18"
input_dir = "PACS/datalists"
image_dir = "PACS/kfold/"
root_path = "outputs/ckpt/res18_pacs"

if not os.path.exists(root_path):
    os.makedirs(root_path)
ckpt_path = root_path + "/" + target + ".tar"
if not os.path.exists(ckpt_path):
    url_path = url[target]
    print(f"Download ckpt {target} from {url_path}")
    wget.download(url_path, out = ckpt_path)
##############################################################################

os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
            f'python test.py '
            f'--source {source[0]} {source[1]} {source[2]} '
            f'--target {target} '
            f'--config {config} '
            f'--ckpt {ckpt_path} '
            f'--input_dir {input_dir} '
            f'--image_dir {image_dir}'
            )
