Simple Multi-Tile Matrix Multiplication (16x16x16)
====================================================

Here I will discuss the simple implentation of tiled matrix multiplication. We split both 16x16 matrices into 4x4 blocks. Within one block, row-major format is used. For matrix A, at the block level, we use row-major access, aswell. For matrix B, we use a block-level column-major scheme.



If you need to calculate larger matrix multiplications within one tile, it is necessary to split the matrices into blocks and/or accumulate partial results. The mmul class provides the ``mmul::mac()`` function with will perform acc = acc + A*B where acc is the (vectorized) values in the accumulator register.

Note that the AIE API docs provide template kernel code for calculating larger matrices within one tile [1]_, but here we give our own simple implementation.

File structure:
::

  .
  ├── graph.hpp # Declaration of the graph
  ├── graph.cpp  # Initializes, runs, and ends the graph. 
  ├── kernels
  │   └── kernels.cpp # matmul kernel implementation
  └── kernels.hpp # declarations of kernel

Kernel Code (kernels.cpp)
****************************

Here is an example of a kernel that performs 4x16x4 matrix multiplication for int16. Note that there is no mmul class to accomodate the larger matrix size. Instead the computation is broken up into 4x4x4 multiplications and partial results are added together using mmul::mac().

.. code-block:: cpp

    #include <aie_api/aie.hpp>
    #include <adf.h>
    using namespace adf;
    using MMUL = aie::mmul<4, 4, 4, int16, int16>;

    void matmul_4x16x4(
        input_buffer<int16>& __restrict a,
        input_buffer<int16>& __restrict b,
        output_buffer<int16>& __restrict c)
    {
        auto a_iter = aie::begin_vector<MMUL::size_A>(a);
        auto b_iter = aie::begin_vector<MMUL::size_B>(b);
        auto c_iter = aie::begin_vector<MMUL::size_C>(c);

        MMUL m;

        // First iteration: initialize accumulator
        m.mul(*a_iter++, *b_iter++);  // m.acc = A0 * B0

        // Subsequent iterations: multiply-accumulate
        for(int i = 1; i < 4; i++) {
            m.mac(*a_iter++, *b_iter++);  // m.acc += A_i * B_i
        }

        // Single store at end
        *c_iter = m.to_vector<int16>();
    }

We can use our 4x16x4 kernel across multiple tiles to calculate a 16x16x16 matrix multiplication in parallel.

Graph Code (graph.h)
*********************

Here is the graph code. Note that this code broadcast the A and B matrices to all tiles. Our inputs are thus duplicated 16 times(!) in local tile memory. We will discuss alternative approaches in the next section, but this serves as a simple reference example.

.. code-block:: cpp

    #include <adf.h>
    #include "kernels.h"
    #include "include.h"

    using namespace adf;

    class MatMulGraph : public graph {
    public:
        input_plio a_in;
        input_plio b_in;
        output_plio c_out[4][4];
        
        input_port a_block_param[4][4];
        input_port b_block_param[4][4];

        kernel mmul[4][4];

        MatMulGraph() {
        a_in = input_plio::create("A_Matrix", plio_128_bits, "data/A_matrix.txt");
            b_in = input_plio::create("B_Matrix", plio_128_bits, "data/B_matrix.txt");

            // Create 4x4 kernel grid
            for(int row=0; row<4; row++) {
                for(int col=0; col<4; col++) {
                    mmul[row][col] = kernel::create(matmul_4x16x4);

                    // Connect A row block (16x16 = 256 elements)
                    connect(a_in.out[0], mmul[row][col].in[0]);
                    dimensions(mmul[row][col].in[0]) = {256}; 

                    // Connect B column block (16x16 = 256 elements)
                    connect(b_in.out[0], mmul[row][col].in[1]);
                    dimensions(mmul[row][col].in[1]) = {256};
                                
                    // Connect parameter ports
                    connect(a_block_param[row][col], mmul[row][col].in[2]);
                    connect(b_block_param[row][col], mmul[row][col].in[3]);

                    // Create PLIO with 32-bit interface for 4x4 int16 blocks
                    c_out[row][col] = output_plio::create(
                    plio_128_bits,
                    "data/C_output_"+std::to_string(row)+"_"+std::to_string(col)+".txt"
                    );

                    connect(mmul[row][col].out[0], c_out[row][col].in[0]);

                    // Set buffer dimensions (16 int16 elements = 4x4 matrix)
                    dimensions(mmul[row][col].out[0]) = {16};

                    // Map to physical tiles
                    location<kernel>(mmul[row][col]) = tile(row, col);
            source(mmul[row][col]) = "src/kernels/kernels.cpp";
            runtime<ratio>(mmul[row][col]) = 1.0;
                }
            }
        }
    }


Host Code (host.cpp)
*********************

The host code calls our graph and sets the hyper-parameters a_block_param and b_block_param. These parameters can be thought of as threadIDs from GPU programming. They allow a kernel to know which block of the larger matrix it should compute.
These parameters are streamed in as initial data packets prior to the other data. These are static parameters that don't change during runtime. We will discuss a way to set these kernel parameters during compile-time in the next module.

.. code-block:: cpp

    #include <adf.h>
    #include "kernels.h"
    #include "graph.h"
    #include "include.h"

    using namespace adf;

    MatMulGraph mmul_graph;

    int main(void) {

        mmul_graph.init();
        for(int row=0; row<4; row++) {
            for(int col=0; col<4; col++) {
            mmul_graph.update(mmul_graph.a_block_param[row][col], row);
            mmul_graph.update(mmul_graph.b_block_param[row][col], col);
            }
        }

        mmul_graph.run(1);
        mmul_graph.end();
        return 0;
    }

AIE Grid Array View
****************************
If we perform software simulation of the above code with Vitis, we are given the following view of the AIE grid.
.. image:: image/16x16x16array.svg
   :alt: Simple multi-tile matmul grid layout
   :width: 100pt
   :align: center

You can see that the 4x4 kernel grid is mapped to the bottom left corner of the AIE grid as specified in the graph code. Note the double buffering which is automatically applied from plio to local tile buffers. If double buffering will exceed the memory limit of the local tiles, you can disable it with ``single_buffer(port<T>&)`` [2]_.


.. rubric:: References
.. [1] AIE API mmul Class. https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__mmul.html