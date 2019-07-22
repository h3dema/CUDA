# Install the NVIDIA display driver on Ubuntu 18

There are a few methods for installing the proprietary NVIDIA display driver. 
For Ubuntu installs the one I recommend is to install the latest driver by adding the the graphics driver ppa repository.
This is well maintained and you will be able to get the latest driver (or older ones if needed).

Before you install the driver it is good to be certain dkms (dynamic kernel module support) and build-essentials is installed. This is because whenever the system kernel is updated dkms will automatically rebuild the NVIDIA driver module for the new kernel. It needs build tools and the kernel headers for this which will be installed with build-essential.

```
sudo apt-get install dkms build-essential
```

The command to add graphics-drivers ppa is shown below. After that you can install the driver,


```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-390
```

This installs the latest driver version 390 (as of this writing). That will work with NVIDIA Volta and lower cards.
This install should be robust. New kernels or drivers should automatically rebuild with dkms.

Now you can reboot!

```
sudo shutdown -r Now
```


# Install CUDA "dependencies"

There are a few dependencies that get installed when you run the CUDA deb file but, since we are not going to use the deb file, you will want to install them separately. It's simple since we can get what's needed with just four package installs,

```
sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev
```

# Get the CUDA "run" file installer

Go to the [CUDA Zone](https://developer.nvidia.com/cuda-zone) and click the Download Now button. 
Then click the link buttons until you get the your configuration.
* Operating system: linux
* Architecture: x86_64
* Distribution: Ubuntu
* Version: 18.04
* Instaler type: runfile (local)

This setup allows you to download

```
wget -c https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
```


# Run the "runfile" to install the CUDA toolkit and samples

This is where we get the CUDA developer toolkit and samples onto the system. Just use sh to run the shell script (runfile),
```
sudo sh cuda_10.1.168_418.67_linux.run
```

You will be asked several questions. Here are my answers, (after accepting the EULA),

`
You are attempting to install on an unsupported configuration. Do you wish to continue?
(y)es/(n)o [ default is no ]: y

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 396.26?
(y)es/(n)o/(q)uit: n

Install the CUDA 9.2 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
 [ default is /usr/local/cuda-9.2 ]:

Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y

Install the CUDA 9.2 Samples?
(y)es/(n)o/(q)uit: y

Enter CUDA Samples Location
 [ default is /home/kinghorn ]: /usr/local/cuda-9.2
 `
 
 
