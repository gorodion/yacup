from IPython import get_ipython
import warnings
from os.path import join
from omegaconf import OmegaConf
import torch
if get_ipython() is None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm
warnings.filterwarnings("ignore")

from utils import Logger

cfg = OmegaConf.load('config.yaml')

# joining paths
for k, v in cfg.rel_paths.items():
    setattr(cfg, k, join(cfg.pro_path, v))

if cfg.debug:
    cfg.sem = 30
    cfg.n_load = 100
    cfg.num_steps = 5
    cfg.n_sample = 150
    cfg.epochs = 1
    cfg.text_encoder_model = 'cointegrated/rubert-tiny'
    cfg.text_tokenizer = 'cointegrated/rubert-tiny'
    cfg.text_embedding = 312

logger = Logger(cfg.logs_path)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
