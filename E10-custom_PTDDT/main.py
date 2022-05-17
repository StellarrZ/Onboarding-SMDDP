import os
import sys
import numpy as np
import torch
from torch import nn
from torch.nn.modules import Module
import torch.distributed as dist


N = 10  # input size, weight size
STEPS = 10
NPDTYPE = np.float32 if "FP32" not in sys.argv else np.float32
DTYPE = torch.float32 if "FP32" not in sys.argv else torch.float32
EPOCHS = 1200


class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.linear = nn.Linear(10, 1, bias=False)
        self.linear.weight = self._init_weights()

    def forward(self, x):
        x = self.linear(x)
        return x

    def _init_weights(self):
        npa = np.asarray([list(range(0, N * STEPS, STEPS))], dtype=NPDTYPE)
        return torch.nn.Parameter(torch.Tensor(npa).type(DTYPE).cuda())


class DistributedDataParallel(Module):
    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._map_grad_mul)
            self._broadcast(param)
    
    def forward(self, x):
        return self.module.forward(x)
    
    def _map_grad_mul(self, grad):
        return 0.01 * grad

    def _broadcast(self, grad, root=0):
        dist.broadcast(grad, root)


def log_weights(model, finalFlag=False, batch_idx=-1):
    weights = model.module.linear.weight.data.cpu().numpy()
    evalScalar = eval(model)
    grads = model.module.linear.weight.grad.cpu().numpy() if not finalFlag else None
    tem = "\n%d/%d  iter %d\n" %(model.rank, model.size, batch_idx) \
            + str(weights) + "\n" + str(evalScalar) + "\n" + str(grads) + "\n"
    with open(log_dir, "a") as fh:
        fh.write(tem)


def train(model, inp, label, optimizer, criterion):
    model.train()
    for batch_idx in range(EPOCHS):
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(torch.sqrt(torch.abs(out)), label)
        loss.backward()
        log_weights(model, batch_idx=batch_idx)
        optimizer.step()
    log_weights(model, finalFlag=True)


def dumb_input():
    rank = dist.get_rank()
    size = dist.get_world_size()
    proportion = rank / (size - 1) - 0.5 if size > 1 else 1
    npa = np.asarray([[i + proportion for i in range(N)]], dtype=NPDTYPE)
    return torch.as_tensor(npa).to(device)


def dumb_label():
    npa = np.asarray([[0.0]], dtype=NPDTYPE)
    return torch.as_tensor(npa).to(device)


def eval(model):
    return torch.mm(model.module.linear.weight.data, dumb_input().t()).cpu().numpy()


def main():
    torch.manual_seed(1)

    model = DistributedDataParallel(Toy().to(device))

    inp = dumb_input()
    label = dumb_label()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)

    train(model, inp, label, optimizer, criterion)



if __name__ == '__main__':
    log_folder = os.path.join(os.getcwd(), "logs")
    os.system("mkdir -p %s" %(log_folder))
    dist.init_process_group(backend="nccl")
    log_dir = os.path.join(log_folder, "log_proc_%02d.txt" %(dist.get_rank()))
    os.system("rm -f %s" %(log_dir))
    device = torch.device("cuda", dist.get_rank())
    main()
    # os.system("cat %s" %(log_dir))
    dist.destroy_process_group()