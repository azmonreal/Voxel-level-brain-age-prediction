from monai.transforms.post.array import AsDiscrete
from monai.transforms.compose import Compose
from monai.transforms.utility.array import EnsureType
from main import *


post_segpred = Compose([EnsureType("tensor", device="cpu")])
post_seglabel = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=4)])
# post_agepred = Compose([EnsureType("tensor", device="cpu")])
post_agelabel = Compose([EnsureType("tensor", device="cpu")])


step_loss_values = []
dicemetric_values = []
maemetric_values = []
globmaemetric_values = []
dice_val_best = 0.0
mae_val_best = 0.0
global_step = 0
global_step_best = 0

eval_num = 1

