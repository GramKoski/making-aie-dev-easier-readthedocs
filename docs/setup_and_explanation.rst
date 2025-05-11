Setup and Architecture Explanation
======================


----------------------
Tooling
----------------------

Before starting our programming we first want to have an idea of the architecture we're targetting and what we want to happen internally. The AMD AI Engine has a unique architecture in which different levels of parallelism that can be utilized. To show how one can bring their own kernels into the architecture, we will first explain the different components of the AI Engine. Then we will show an example for a simple AIE vector addition and how the different but similar code synthesizes onto the actual target architecture.

The AI Engine suite of tools includes a compiler and both a software and hardware emulator. Vitis itself includes features to trace signals, generate dataflow graphs, show data movement over hardware and cycle time, profile the kernel and io, and much more. While the process of compiling and emulation can be done from command line tools, to use more advanced analysis will require Vitis IDE. All of these tools are freely available for download by AMD. You will also need an AMD AIE `build license <https://www.xilinx.com/getlicenses>`_.

----------------------
Background
----------------------

The AI Engine is a technology developed by Xilinx and then adopted by AMD after a finalized Xilinx acquisition on Valentines Day 2022 (February 14th). [#AIE-Tech]_ [#Xilinx-Acquisition]_ It's successor, the AI Engine-ML (AIE-ML) makes a few changes to the architecture and is rebranded as an NPU. This new architecture is referred to as XDNA. [#XDNA-arch]_ 

The AMD AI Engine has gone through a few iterations that change the support and architecture. As of May 2025, there are currently 3 iterations of the AIE. Confusingly, because the term *AIE* refers to a technology and the first generation of *AIE*, sometimes the AIE-ML or onwards is also referred to as an AIE. For the sake of clarity, we will refer to the AIE technology as the *AIE*, and will refer to specific versions as:

1. **AIE1**: First generation of the AIE. For clarity we refer it to AIE1, and not simply as AIE.
2. **AIE-ML**: Addition of an AIE Memory Tile. These makes up XDNA.
3. **AIE-MLv2**: Second generation of AIE-ML. These make up XDNA2.

The AIE is integrated AMD's CPUs (referred to as the *Ryzen AI* Series) and AMD's certain *Versal* FPGAs. It has kept its name through the Xilinx acquisition, as *Versal* was originally given and planned by Xilinx way back in 2018. [#unveiled-versal]_ Because of the only certain Versal SoCs include the AIE, we include a small list of their products here. We also make an important make an important distinction that AMD's Alveo Accelerator cards are not an addition to this list, as the Alveo V80 and Alveo V70 are Versal SoCs, hence the **V** in the naming convention. [#Alveo-p-guide]_


**First Generation Versal SoC Series Table**

+------------+------------+------------+-----------+-----------+
| Series     | Products   | AIE-Type   | Number    | Notes     |
+============+============+============+===========+===========+
| Versal     | VPK120     | N/A        | 0         | Eval Kit  |
| Premium    | VPK180     |            |           |           |
|            +------------+------------+-----------+-----------+
|            | VP2502     | AIE1       | 472       |           |
|            +------------+            |           |           |
|            | VP2802     |            |           |           |
|            +------------+------------+-----------+           |
|            | VP1002     | N/A        | 0         |           |
|            | VP1052     |            |           |           |
|            | VP1102     |            |           |           |
|            | VP1202     |            |           |           |
|            | VP1402     |            |           |           |
|            | VP1502     |            |           |           |
|            | VP1552     |            |           |           |
|            | VP1702     |            |           |           |
|            | VP1802     |            |           |           |
|            | VP1902     |            |           |           |
+------------+------------+------------+-----------+-----------+
|            | VEK280     | AIE-ML     | 304       | Eval Kit  |
| Versal     +------------+------------+-----------+-----------+
| AI Edge    | VE2002     | AIE-ML     | 8         |           |
|            +------------+            +-----------+           +
|            | VE2102     |            | 12        |           |
|            +------------+            +-----------+           +
|            | VE2202     |            | 24        |           |
|            +------------+            +-----------+           +
|            | VE2302     |            | 34        |           |
|            +------------+            +-----------+           +
|            | VE2602     |            | 152       |           |
|            +------------+            +-----------+           +
|            | VE2802     |            | 304       |           |
|            +------------+------------+-----------+           +
|            | VE1752     | AIE1       | 304       |           |
+------------+------------+------------+-----------+-----------+
|            | VR1602     | AIE1       | 126       |           |
| Versal     +------------+            +           +           +
| RF         | VR1652     |            |           |           |
|            +------------+            +-----------+           +
|            | VR1902     |            | 120       |           |
|            +------------+            +           +           +
|            | VR1952     |            |           |           |
+------------+------------+------------+-----------+-----------+
| Versal     | VCK190     | AIE1       | 400       | Eval Kit  |
| AI Core    +------------+------------+-----------+           +
|            | VC1502     | AIE1       | 198       |           |
|            +------------+------------+-----------+           +
|            | VC1702     | AIE1       | 304       |           |
|            +------------+------------+-----------+           +
|            | VC1802     | AIE1       | 300       |           |
|            +------------+------------+-----------+           +
|            | VC1902     | AIE1       | 400       |           |
|            +------------+------------+-----------+           +
|            | VC2602     | AIE-ML     | 152       |           |
|            +------------+------------+-----------+           +
|            | VC2802     | AIE-ML     | 304       |           |
+------------+------------+------------+-----------+-----------+
|            | VMK180     | N/A        | 0         | Eval kit  |
| Versal     +------------+------------+-----------+-----------+
| Prime      | VM1102     | N/A        | 0         |           |
|            | VM1302     |            |           |           |
|            | VM1402     |            |           |           |
|            | VM1502     |            |           |           |
|            | VM1802     |            |           |           |
|            | VM2152     |            |           |           |
|            | VM2202     |            |           |           |
|            | VM2302     |            |           |           |
|            | VM2502     |            |           |           |
|            | VM2902     |            |           |           |
+------------+------------+------------+-----------+-----------+
|            | VHK158     | N/A        | 0         | Eval kit  |
| Versal     +------------+------------+-----------+-----------+
| HBM        | VM1102     | N/A        | 0         |           |
|            | VM1302     |            |           |           |
|            | VM1402     |            |           |           |
|            | VM1502     |            |           |           |
|            | VM1802     |            |           |           |
|            | VM2152     |            |           |           |
|            +------------+------------+-----------+-----------+
|            | XCV80      | N/A        | 0         | Alveo V80 |
+------------+------------+------------+-----------+-----------+
[#versal]_

**Second Generation Versal SoC Series Table**

+------------+------------+------------+-----------+----------------+
| Series     | Products   | AIE-Type   | Number    | Notes          |
+============+============+============+===========+================+
| Versal     | 2VP3102    | N/A        | 0         |                |
| Premium    | 2VP3202    |            |           |                |
| Gen 2      | 2VP3402    |            |           |                |
|            | 2VP3602    |            |           |                |
+------------+------------+------------+-----------+----------------+
|            | 2VE3304    | AIE-MLv2   | 24        |                |
| Versal     +------------+            |           |                |
| AI Edge    | 2VE3358    |            |           |                |
| Gen 2      +------------+            +-----------+                |
|            | 2VE3504    |            | 96        |                |
|            +------------+            |           |                |
|            | 2VE3558    |            |           |                |
|            +------------+            +-----------+                |
|            | 2VE3804    |            | 144       |                |
|            +------------+            |           |                |
|            | 2VE3858    |            |           |                |
+------------+------------+------------+-----------+----------------+
| Versal     | 2VM3358    | N/A        | 0         |                |
| Prime      | 2VM3558    |            |           |                |
| Gen 2      | 2VM3654    |            |           |                |
|            | 2VM3858    |            |           |                |
+------------+------------+------------+-----------+----------------+
| Versal     | ?          | AIE-MLv2   | 0         | Discontinued   |
| AI Core    |            |            |           |[#v70-product]_ |
| (Gen 2?)   |            |            |           |                |
|            |            |            |           |                |
+------------+------------+------------+-----------+----------------+
[#versal]_

*Again, other Alveo cards are not included since they are not Versal*. 


CPUs are much easier. AIE in this context is always an NPU, from either architecture XDNA or XDNA2. In other words either AIE-ML or AIE-MLv2. 

AMD makes this list of AMD Ryzen AI CPUs so much easier to find. The generations to have NPUs are Ryzen 7040 (*"Phoenix"*), Ryzen 8040 (*"Hawk Point"*), and Ryzen 300 (*"Strix Point*"). [#adv-ai-keynote]_ You can view a complete list of processors that have an NPU at `AMD Ryzen AI <https://www.amd.com/en/products/processors/consumer/ryzen-ai.html>`_. 

+---------------+------------+-----------+----------------+
| Series        | AIE-Type   | Number    | Notes          |
+===============+============+===========+================+
| Ryzen AI 7040 | AIE-ML     | 16        |                |
| (Phoenix)     | (XDNA)     |           |                |
+---------------+            |           |                |
| Ryzen AI 8040 | AIE-ML     |           |                |
|(Hawk Point)   | (XDNA)     |           |                |
+---------------+------------+-----------+----------------+
| Ryzen AI 300  | AIE-MLv2   | 32        | (Including     |
|               | (XDNA2)    |           | Ryzen AI       |
| (Strix Point) |            |           | Max)           |
+---------------+------------+-----------+----------------+


----------------------
Architecture Differences
----------------------

Since the naming scheme does little distinguate the differences accross versions, we've dedicated a section to explain the differences.


+-----------------------+------------+------------+-----------+
|      Feature          | AIE1       | AIE-ML     | AIE-MLv2  |
+=======================+============+============+===========+
|      Memory Tile      | Does not   | column 2   | column 3  |
|                       | exist      |            |           |
+-----------------------+------------+------------+-----------+
|                       | body row 2 | Cells may span columns.|
+-----------------------+------------+------------+-----------+
|                       | body row 3 | Cells may  | - Cells   |
+-----------------------+------------+ span rows. | - contain |
|                       | body row 4 |            | - blocks. |
+-----------------------+------------+------------+-----------+
[#bltinc-diff]_

*Note: the AIE and the AIE-ML have been referred to as AIE-1 and AIE-2. This is outdated convention, so please don't use this*


.. [#AIE-Tech] https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/ai-engine.html
.. [#Xilinx-Acquisition] https://ir.amd.com/news-events/press-releases/detail/1047/amd-completes-acquisition-of-xilinx
.. [#XDNA-arch] https://www.amd.com/en/technologies/xdna.html
.. [#bltinc-diff] https://bltinc.com/2025/03/19/amd-versal-ai-engine-and-ai-engine-ml-which-is-right/
.. [#Alveo-p-guide]   https://docs.amd.com/v/u/en-US/alveo-product-selection-guide
.. [#versal] https://www.amd.com/en/products/adaptive-socs-and-fpgas/versal.html
.. [#unveiled-versal] https://www.prnewswire.com/news-releases/xilinx-unveils-versal-the-first-in-a-new-category-of-platforms-delivering-rapid-innovation-with-software-programmability-and-scalable-ai-inference-300721588.html
.. [#v70-product] https://www.prnewswire.com/news-releases/xilinx-unveils-versal-the-first-in-a-new-category-of-platforms-delivering-rapid-innovation-with-software-programmability-and-scalable-ai-inference-300721588.html
.. [#adv-ai-keynote] https://www.amd.com/content/dam/amd/en/documents/advancing-ai-keynote.pdf#page=79
.. [#Ryzen-AI] https://www.amd.com/en/products/processors/consumer/ryzen-ai.html
