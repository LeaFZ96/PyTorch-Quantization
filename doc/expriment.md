# Experiment

## Usage of NVIDIA Docker

Pull the PyTorch Image

```bash
$ sudo docker pull nvcr.io/nvidia/pytorch:19.06-py3
```

1. **Interactive mode:** Open a command prompt and issue:

   ```
   nvidia-docker run -it --rm -v local_dir:container_dir 
   nvcr.io/nvidia/pytorch:<xx.xx>-py3
   ```

2. **Non-interactive mode:** Open a command prompt and issue:

   ```
   nvidia-docker run --rm -v local_dir:container_dir 
   nvcr.io/nvidia/pytorch:<xx.xx>-py3 <command>
   ```

## VGG16 on CIFAR10

1. Run the interactive mode:

   ```bash
   $ sudo nvidia-docker run -it --rm -v /home/leafz/PyTorch-Learning:/workspace nvcr.io/nvidia/pytorch:19.06-py3
   ```

2. Run the Non-interactive mode:

   ```bash
   $ sudo nvidia-docker run --rm -v /home/leafz/PyTorch-Learning:/workspace nvcr.io/nvidia/pytorch:19.06-py3 python /workspace/src/vgg_p.py
   ```

Both of the above commands meet the error:

```bash
NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
   insufficient for PyTorch.  NVIDIA recommends the use of the following flags:
   nvidia-docker run --ipc=host ...

Files already downloaded and verified
Let's use 4 GPUs!
start train

Traceback (most recent call last):
  File "/workspace/src/vgg_p.py", line 58, in <module>
    outputs = net(inputs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 494, in __call__
    result = self.forward(*input, **kwargs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py", line 152, in forward
    outputs = self.parallel_apply(replicas, inputs, kwargs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py", line 162, in parallel_apply
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/parallel/parallel_apply.py", line 83, in parallel_apply
    raise output
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/parallel/parallel_apply.py", line 59, in _worker
    output = module(*input, **kwargs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 494, in __call__
    result = self.forward(*input, **kwargs)
  File "/opt/conda/lib/python3.6/site-packages/torchvision/models/vgg.py", line 44, in forward
    x = self.classifier(x)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 494, in __call__
    result = self.forward(*input, **kwargs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/container.py", line 92, in forward
    input = module(input)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 494, in __call__
    result = self.forward(*input, **kwargs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 92, in forward
    return F.linear(input, self.weight, self.bias)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/functional.py", line 1403, in linear
    ret = torch.addmm(bias, input, weight.t())
RuntimeError: size mismatch, m1: [400 x 512], m2: [25088 x 4096] at /tmp/pip-req-build-hlju8y6w/aten/src/THC/generic/THCTensorMathBlas.cu:273
```

Then I add the `—-ipc=host` to the commands:

```bash
$ sudo nvidia-docker run -it --ipc=host --rm -v /home/leafz/PyTorch-Learning:/workspace nvcr.io/nvidia/pytorch:19.06-py3
```

```bash
$ sudo nvidia-docker run --ipc=host --rm -v /home/leafz/PyTorch-Learning:/workspace nvcr.io/nvidia/pytorch:19.06-py3 python /workspace/src/vgg_p.py
```

Still have the error. Noticed that this scipt can be successfully run outside.

After reading the source code of vgg model: `python3.6/site-packages/torchvision/models/vgg.py`, I found the cause of the problem.

In the docker, python use `torchvision 0.2.1`, but the host is `torchvision 0.3.0`. There is a little difference between the two implement of the vgg model:

```python
# vgg.py of torchvision 0.3.0

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return 
      
# vgg.py of torchvision 0.2.1

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return 
```

An `avgpool` added in the new version, it make the output of the `self.features(x)` from the size `([batch_size, 512, *, *])` to `([batch_size, 512, 7, 7])`. So that it can fit the input size of the `self.classifier(x)` after the flat by `x.view(x.size(0), -1)`.

In my script, I used the CIFAR10 dataset, which data size is `([batch_size, 3, 32, 32])`. It need to be resize to `([batch_size, 512, 7, 7])`, or will raise the above error.

To solve the problem, update the torchvision in the docker:

```bash
$ conda install torchvision -c pytorch
```

Then, the script can be run correctly. But it's too slow in the docker.

## resnet50 on ImageNet

### Host

**Single GPU**

```bash
$ python main.py -a resnet50 --epochs 1 --gpu 0 ../../imagenet
```

Parameters: `batch-size: 256`, `workers: 4`, `lr: 0.01`

Result in the `log/resnet_s_256.log`

```bash
$ python main.py -a resnet50 --epochs 1 --batch-size 128 --gpu 0 ../../imagenet
```

Parameters: `batch-size: 128`, `workers: 4`, `lr: 0.01`

Result in the `log/resnet_s_128.log`

**Data Parallel**

```bash
$ python main.py -a resnet50 --epochs 1 ../../imagenet
```

Parameters: `batch-size: 256`, `workers: 16`, `lr: 0.01`

Result in `log/resnet_p_256.log` 

```bash
$ python main.py -a resnet50 --epochs 1 --batch-size 1024 --workers 16 ../../imagenet
```

Parameters: `batch-size: 1024`, `workers: 16`, `lr: 0.01`

Result in `log/resnet_p_1024.log`

**Distributed Data Parallel (Single node, 4 GPUs)**

```bash
$ python main.py -a resnet50 --epochs 1 --batch-size 1024 --workers 16 --dist-url 'tcp://127.0.0.1:2345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ../../imagenet
```

Parameters: `batch-size: 1024`, `workers: 16`, `lr: 0.01`

Result in `log/resnet_d_1024.log`

```bash
$ python main.py -a resnet50 --epochs 1 --batch-size 256 --dist-url 'tcp://127.0.0.1:2345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ../../imagenet
```

Parameters: `batch-size: 256`, `workers: 4`, `lr: 0.01`

Result in `log/resnet_d_256.log`

####Some thoughts about the results

When I set a small batch size $a$ and a number of iteration $b$ , versus a large batch size $c$ and a number of iteration $d$ where $ab = cd$ . I find that after being trained for a same number of ephoc, the model with small batch size always performs better and the model with large batch size could be trained faster. I found a paper [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) and a question [Tradeoff batch size vs. number of iterations to train a neural network](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network) about that.

The above thoughts of the batch size are about the training on a single GPU. As for single GPU versus multiple GPUs, there are something should be noticed. The following picture is about how `DataParallel` works. 

![dataparallel](img/e153c9fa07a2688c77fc1d7a572f26d12828528d_2_275x500.png)

The summary of each step is:

1. Default GPU split the batch to the four part and scatter to the other GPUs
2. Each GPU copy the model from the default GPU
3. Each GPU forward the input and get the output
4. Gather the output to the default GPU. Default GPU compute the respective loss and scatter the losses to the other GPUs
5. Each GPU computer their gradients
6. Default GPU sum up the gradients and then update the model

Because the gradient of the multiple GPUs is the sum of the each GPU's gradient, The training model on the multiple GPUs should compare with the model with same batch size on the single GPU. So the steps which can be speed up of the training procedure are the forward compute and gradient compute. Under this condition, the speedup depends on the proportion of the above two steps in the entire process.

For example, we can compute the approximate speedup of the 4 GPUs with `batch-size=256`:
$$
speedup = \frac{0.742}{0.378} = 1.963
$$
It's just a speedup of `DataParallel` implementation by PyTorch in **4** GPUs.

### NVIDIA Docker

