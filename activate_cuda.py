import sys
import torch

if 'gpu' in sys.argv[1]:
    numGPU = [[int(s) for s in re.findall(r'\d+', sys.argv[1])][-1]]
    cuda_str = ""
    for i in range(len(numGPU)):
        cuda_str = cuda_str + str(numGPU[i])
        if i is not len(numGPU) - 1:
            cuda_str = cuda_str + ","
    print("Devices to use:", cuda_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str

print(cuda_str)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(sys.argv)
print(device)