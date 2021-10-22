from IPython import get_ipython
import logging
import warnings
from os.path import join
import torch

from utils import Logger

if get_ipython() is None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm
logging.basicConfig(filename='stuff.log', encoding='utf-8', level=logging.DEBUG)
warnings.filterwarnings("ignore")

debug = False

# download parameters
with_async = True
sem = 1000
n_load = 30
min_id = 0
total_images = 5462418 # or None

# paths
PRO_PATH = 'E:/yacup'
image_urls_path = join(PRO_PATH, 'images.json') # TODO relpath via yaml
captions_path = join(PRO_PATH, 'meta.json')
images_path = join(PRO_PATH, 'images')
save_path = join(PRO_PATH, 'last.pt')
logs_path = join(PRO_PATH, 'bad.log')
val_data_path = join(PRO_PATH, 'val/datasets')
val_gt_path = join(PRO_PATH, 'val/gt.json')

# image size
size = 256

# training parameters
batch_size = 32
num_steps = 5000
n_sample = 50_000
num_workers = 1
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 2
factor = 0.8
epochs = 100

# models parameters
model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-multilingual-cased"
text_embedding = 768
text_tokenizer = "distilbert-base-multilingual-cased"
max_length = 200
pretrained = True  # for both image encoder and text encoder
trainable = True  # for both image encoder and text encoder
temperature = 1.0

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256
dropout = 0.1

logger = Logger(logs_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")