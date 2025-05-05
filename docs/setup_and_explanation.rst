Setup and Architecture Explanation
======================


----------------------
Tooling
----------------------

Before starting our programming we first want to have an idea of the architecture we're targetting and what we want to happen internally. The AMD AI Engine has a unique architecture in which different levels of parallelism that can be utilized. To show how one can bring their own kernels into the architecture, we will first explain the different components of the AI Engine. Then we will show an example for a simple AIE vector addition and how the different but similar code synthesizes onto the actual target architecture.

The AI Engine suite of tools includes a compiler and both a software and hardware emulator. Vitis itself includes features to trace signals, generate dataflow graphs, show data movement over hardware and cycle time, profile the kernel and io, and much more. While the process of compiling and emulation can be done from command line tools, to use more advanced analysis will require Vitis IDE. All of these tools are freely available for download by AMD. You will also need an AMD AIE `build license <https://www.xilinx.com/getlicenses>`_.

.. image:: magnetic-balls.jpg
   :width: 40pt



----------------------
Architecture
----------------------
