import torch
from tqdm import tqdm
from torch import tensor as T
import numpy as np

def prepare_for_distributed(args, model):
    assert torch.cuda.is_available()
    assert torch.cuda.device_count()>1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device = args.local_rank)
    return model

def distributed_load_data(data, local_rank, distributed,drop_last = True):
    samples = []
    if distributed:
        world_size = torch.distributed.get_world_size()
        if drop_last:
            data = data[:len(data)//world_size*world_size] # drop last 효과
        else:
            num_samples = math.ceil(len(data)/world_size)
            total_size = num_samples*world_size
            padding_size = total_size - num_samples
            if padding_size <= len(data):
                data += data[:padding_size]
            else:
                data += (data*math.ceil(padding_size/len(data)))[:padding_size] 
        num_samples = math.ceil(len(data)/world_size)
        samples = data[local_rank:local_rank+num_samples]
        return samples
    return data

def get_global(args, thing):
    to_gather = [torch.zeros_like(thing) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list = to_gather, tensor = thing)
    global_thing = []
    for j in range(torch.distributed.get_world_size()):
        if j!=args.local_rank:  
            global_thing.extend(to_gather[j].to(thing.device))
        else:
            global_thing.extend(thing)
    return global_thing

def gather_tensors(input_array):
    # multi dimension이라면 1d로 진행함
    world_size = torch.distributed.get_world_size()
    ## gather shapes first
    myshape = input_array.shape
    mycount = torch.prod(torch.tensor(myshape)) # 다 곱함.
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)]
    torch.distributed.all_gather(all_shape, shape_tensor)
    ## compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    ## padding tensors and gather them
    output_tensors = [torch.Tensor(max_count).cuda() for i in range(world_size)]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    torch.distributed.all_gather(output_tensors, input_tensor)
    ## unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [x[:all_count[i]].reshape(all_shape[i]) for i,x in enumerate(padded_output)]
    return output 