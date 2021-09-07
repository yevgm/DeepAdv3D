import torch
import numpy as np
from utils.fs import convert_bytes
import random



def identify_system():
    from platform import python_version
    from utils.torch.cpuinfo import cpu
    import psutil
    print(f'Python {python_version()} , Pytorch {torch.__version__}, CuDNN {torch.backends.cudnn.version()}')
    cpu_dict = cpu.info[0]
    mem = psutil.virtual_memory().total
    num_cores_str = f" :: {psutil.cpu_count() / psutil.cpu_count(logical=False)} Cores"
    mem_str = f" :: {convert_bytes(mem)} Memory"

    if 'ProcessorNameString' in cpu_dict:  # Windows
        cpu_name = cpu_dict['ProcessorNameString'] + num_cores_str + mem_str
    elif 'model name' in cpu_dict:  # Linux
        cpu_name = cpu_dict['model name'] + num_cores_str + mem_str
    else:
        raise NotImplementedError

    print(f'CPU : {cpu_name}')
    gpu_count = torch.cuda.device_count()
    print(f'Found {gpu_count} GPU Devices:')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'\tGPU {i}: {p.name} [{p.multi_processor_count} SMPs , {convert_bytes(p.total_memory)} Memory]')


def set_determinsitic_run(run_config, seed=None):
    if seed is None:
        # Specific to the ShapeCompletion platform
        seed = run_config['UNIVERSAL_RAND_SEED']

    # CPU Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU Seeds
    torch.cuda.manual_seed_all(seed)
    # CUDNN Framework
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Might be best to turn off benchmark for deterministic results:
    # https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
    # dir = r'C:\Users\idoim\Desktop\ShapeCompletion\results'
    # t = TensorboardSupervisor()
    # time.sleep(10)
    # list_of_proccesses = findProcessIdByName('tensorboard.main')
    # print(list_of_proccesses)
    # for process in list_of_proccesses:
    #     os.kill(process['pid'], signal.SIGTERM)
    # t.finalize()
    # print('DONE!')
    # import torch
    # import torchvision
    #
    # model = torchvision.models.resnet50(False)
    # pymodel = PytorchNet.monkeypatch(model)
