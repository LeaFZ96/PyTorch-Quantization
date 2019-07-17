

# Envionment Setting


## Prepare data

Mount information:

```bash
$ findmnt -lo source,target,fstype,label,options,used -t ext4
SOURCE    TARGET FSTYPE LABEL OPTIONS                                      USED
/dev/sda2 /      ext4         rw,relatime,errors=remount-ro,data=ordered  33.8G
/dev/md0  /home  ext4         rw,relatime,discard,stripe=256,data=ordered  1.1T
```

`/home` already mounted on the `/dev/md0`. Create a soft link to the **ImageNet** data:

```bash
$ ln -s /home/saurabh/imagenet_data /home/leafz/
```

## Updata the driver

Add the Proprietary GPU Drivers PPA source

```bash
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
```

Remove the old nvidia packages
```bash
$ sudo apt remove nvidia*
$ sudo apt autoremove
```

Install new driver (418)
```bash
$ sudo ubuntu-drivers autoinstall
```

Reboot
```bash
$ sudo reboot
```

New driver information:

```bash
$ nvidia-smi
Tue Jul 16 12:59:51 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:1A:00.0 Off |                    0 |
| N/A   29C    P0    40W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:1B:00.0 Off |                    0 |
| N/A   30C    P0    41W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:3D:00.0 Off |                    0 |
| N/A   31C    P0    56W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:3E:00.0 Off |                    0 |
| N/A   28C    P0    56W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Install Docker

### Set up the repository

1. Update the `apt` package index:

   ```bash
   $ sudo apt-get update
   ```

2. Install packages to allow `apt` to use a repository over HTTPS:

   ```bash
   $ sudo apt-get install \
       apt-transport-https \
       ca-certificates \
       curl \
       gnupg-agent \
       software-properties-common
   ```

3. Add Docker’s official GPG key:
   ```
   $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   ```

   Verify that you now have the key with the fingerprint `9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88`, by searching for the last 8 characters of the fingerprint.

   ```bash
   $ sudo apt-key fingerprint 0EBFCD88
       
   pub   rsa4096 2017-02-22 [SCEA]
         9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
   uid           [ unknown] Docker Release (CE deb) <docker@docker.com>
   sub   rsa4096 2017-02-22 [S]
   ```

4. Use the following command to set up the **stable** repository.

   ```bash
   $ sudo add-apt-repository \
      "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"
   ```

### Install Docker CE

1. Update the `apt` package index.

   ```bash
   $ sudo apt-get update
   ```

2. Install the *latest version* of Docker CE and containerd, or go to the next step to install a specific version:

   ```bash
   $ sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

## Install NVIDIA Docker

Meet the following prerequistes:

1. GNU/Linux x86_64 with kernel version > 3.10
2. Docker >= 1.12
3. NVIDIA GPU with Architecture > Fermi (2.1)
4. [NVIDIA drivers](http://www.nvidia.com/object/unix.html) ~= 361.93 (untested on older versions)

Add the package repositories

```bash
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
```

Install nvidia-docker2 and reload the Docker daemon configuration

```bash
$ sudo apt-get install -y nvidia-docker2
$ sudo pkill -SIGHUP dockerd
```

