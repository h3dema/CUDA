# Install the NVIDIA display driver on Ubutn 18

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
