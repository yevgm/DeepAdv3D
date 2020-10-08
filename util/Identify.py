import socket
import platform, subprocess, psutil
import os, platform, subprocess, re
import torch
def identify_system():
    PC_name = socket.gethostname()
    OS =  platform.platform(aliased=0, terse=0)
    python_ver = platform.python_version()
    CPU_name, cpu_core = get_processor_name()
    logical_cores = psutil.cpu_count(logical=True)
    vram = psutil.virtual_memory()[0]/(1024.0 ** 3)

    # find current working dir free space
    df = subprocess.Popen(["df", __file__], stdout=subprocess.PIPE)
    output = df.communicate()[0]
    _, _, _, available, _, mountpoint = \
        output.decode("ascii").split("\n")[1].split()
    available = int(available)/(1024.0 ** 2)
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda
    cuDNN_ver = torch.backends.cudnn.version()

    gpu_num = torch.cuda.device_count()
    if gpu_num:
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = ""

    print_all(PC_name,OS,python_ver,CPU_name,cpu_core,\
          logical_cores,vram,mountpoint,available,\
          torch_ver,cuda_ver,cuDNN_ver,gpu_num,gpu_name)


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.decode("ascii").split("\n"):
            if "model name" in line:
                model = re.sub( ".*model name.*:", "", line,1)
            if "cpu cores" in line:
                cores = re.sub(".*cpu cores.*:", "", line, 1)
                return model, cores
    return ""

def print_all(PC_name,OS,python_ver,CPU_name,cpu_core,\
                logical_cores,vram,mountpoint,available,\
                torch_ver,cuda_ver,cuDNN_ver,gpu_num,gpu_name):

    print("Computer Name\t:\t" + PC_name + "\n")
    print("OS\t:\t" + OS + "\n")
    print("Python\t:\t" + python_ver + "\n")
    print("CPU\t:\t" + CPU_name + " [" + str(cpu_core) + " PCores ,"\
                    + str(logical_cores) + " LCores , {:.2f} GB".format(vram)  + \
                    " VRAM]" + "\n")
    print("Free Disk Space\t:\t" + mountpoint + " {:.2f}".format(available) \
          + " GB" + "\n")
    print("Compute Packages\t:\t" "PyTorch " + str(torch_ver) + " CUDA "\
          + str(cuda_ver) + " CuDNN " + str(cuDNN_ver) + "\n")
    print("#Visible GPUs\t:\t" + str(gpu_num) + "\n")

    # TODO: finish this if statement when you know the output of gpu_name
    if gpu_num:
        for i, gpu in gpu_name:
            print("GPU " + str(i) + ": " + gpu + " [ ")
