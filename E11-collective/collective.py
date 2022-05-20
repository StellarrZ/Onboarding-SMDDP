import os
import torch
from torch import distributed as dist

ROOT = 0
TENSOR_LEN = 16
LOG_PATH = os.path.join(os.getcwd(), "logs")



def test_resuelt(result, method="Unknown"):
    answer = torch.tensor(range(TENSOR_LEN), dtype=dtype).to(device) * (size - 1) * size // 2
    ruler = torch.tensor([True] * TENSOR_LEN).to(device)
    assert torch.sum(torch.logical_xor(ruler, result == answer)) == 0
    with open(log_dir, 'a') as fh:
        fh.write("Method %s -- Result Test Passed. Rank %d/%d\n" %(method, rank, size))


def sharing_all_to_all(inp):
    commList = [i for i in range(size) if i != rank]
    buffer = torch.zeros(TENSOR_LEN, dtype=dtype).to(device)
    ret = inp.clone()
    for objRank in commList:
        handle = dist.isend(inp, objRank)
        dist.recv(buffer, objRank)
        ret += buffer
        handle.wait()
    
    return ret


def run_aggregator(inp):
    if rank == ROOT:
        commList = [i for i in range(size) if i != ROOT]
        bufferList = [torch.zeros(TENSOR_LEN, dtype=dtype).to(device) for _ in range(len(commList))]
        ret = inp.clone()
        handleList = []
        for i, src in enumerate(commList):
            handleList.append(dist.irecv(bufferList[i], src))
        for i, handle in enumerate(handleList):
            handle.wait()
            ret += bufferList[i]

        handleList = []
        for dst in commList:
            handleList.append(dist.isend(ret, dst))
        for handle in handleList:
            handle.wait()
    else:
        handle = dist.isend(inp, ROOT)
        ret = torch.zeros(TENSOR_LEN, dtype=dtype).to(device)
        handle.wait()
        dist.recv(ret, ROOT)
    
    return ret


def main():
    inp = torch.tensor(range(TENSOR_LEN), dtype=dtype).to(device) * rank

    out = sharing_all_to_all(inp)
    test_resuelt(out, method="sharing_all_to_all")

    out = run_aggregator(inp)
    test_resuelt(out, method="aggregator")



if __name__ == '__main__':
    os.system("rm -rf %s" %(LOG_PATH))
    os.system("mkdir -p %s" %(LOG_PATH))
    dtype = torch.int

    dist.init_process_group("nccl")
    size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda")
    torch.cuda.set_device(rank)
    log_dir = os.path.join(LOG_PATH, "log_shared.txt")
    # log_dir = os.path.join(LOG_PATH, "log_proc_%02d.txt" %(rank))

    main()
    dist.destroy_process_group()