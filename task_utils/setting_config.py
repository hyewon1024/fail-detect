import torch
from env import set_env_dataCollection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = set_env_dataCollection.make_env()
