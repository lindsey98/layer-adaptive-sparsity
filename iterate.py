import torch,argparse,random,os
import numpy as np
from tools import *

""" ARGUMENT PARSING """
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--cuda', type=int, default=1, help='cuda number')
parser.add_argument('--model', type=str, default='resnet101', help='network')
parser.add_argument('--weights_path', type=str, default='dicts/resnet101_1.pth', help='weights path')
parser.add_argument('--pruner', type=str, default='lamp', help='pruning method')
parser.add_argument('--iter_start', type=int, default=1, help='start iteration for pruning')
parser.add_argument('--iter_end', type=int, default=1, help='start iteration for pruning')

args = parser.parse_args()

""" SET THE SEED """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

DEVICE = args.cuda if args.cuda > 0 else 'cpu'

""" IMPORT LOADERS/MODELS/PRUNERS/TRAINERS"""
model,amount_per_it,batch_size,opt_pre,opt_post = model_and_opt_loader(args.model,DEVICE,args.weights_path)
train_loader, test_loader = dataset_loader(args.model,batch_size=batch_size)
pruner = weight_pruner_loader(args.pruner)
trainer = trainer_loader()

""" SET SAVE PATHS """
DICT_PATH = f'./dicts/{args.model}/{args.seed}'
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)
BASE_PATH = f'./results/iterate/{args.model}/{args.seed}'
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

acc, _ = train.test(model, test_loader)
print("Testing acc before pruning:", acc)

""" PRUNE AND RETRAIN """
for it in range(args.iter_start,args.iter_end+1):
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    pruner(model, amount_per_it)
    acc, _ = train.test(model, test_loader)
    sparsity = get_model_sparsity(model)
    torch.save(model.state_dict(), os.path.join(DICT_PATH, 'pruned.pth'))
    print("Testing acc after pruning:", acc)
    print("% weights left:", sparsity)