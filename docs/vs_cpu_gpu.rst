Comparisons of an NPU Architecture
============================


The AIE architectures were designed with Machine Learning in mind. Even now, most of these applcations are run on GPUs. So why would one prefer to run ML tasks on an AIE? And how would the performance compare?

We'll start off with a comparison of the architectures. 

CPUs are usually designed for low-latency and general usage. With this goal in mind, CPUs usually have multilayered memory hierarchy, anticipate instructions with branch prediction and lookahead execution, can begun to run instructions despite a stall with out-of-order execution or simultaneous multithreading, and will divvy up the instructions can execute at at time within a core's units (superscaling execution) or between cores (multicore). A CPU will employ all these techniques to lower latency and make sure the CPU is fed with instructions to run, even if that means executing a different process.

A GPU places a lower emphasis on latency and instead on thoroughput. This is already evident by the lower clock speeds GPUs normally have when compared to CPUs. While CPUs have a couple of cores on consumer end systems and hundreds on server systems, GPUs usually have thousands of units dedicate for processing. *Example*: The `NVIDIA GeForce RTX 4080 <https://www.techpowerup.com/gpu-specs/geforce-rtx-4080.c3888>`_ has 9728 Cuda Cores (Shading Units) and the `Radeon RX 7900 XTX <https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941>`_ has 6144 Streaming Processors (Shading Units). These units are fundamentally much simpler that CPU cores, yet the amount of them allows for a very large amount of data to be fed through and computed. Additionally, the memory hierarchy differs a bit as well. Though there is still a memory hierarchy, there is usually some larger memory size on the lowest level on chip. On the GeForce RTX 4080 this is 128KB->64MB->16GB and on Radeon RX 7900 XTX 64KB->256KB->6MB->96MB->24GB.


But the AI Engine is very different. For starters, the memory is divided up very differently. Instead of a hierarchal memory, it's more spatial.
