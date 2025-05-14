Matrix Multiplication Example
=============================
The AIE API is a C++ library which provides a convenient set of functions and classes for programming kernels. We will discuss the use of this API in order to perform matrix multiplication.

Single Tile Example
-------------------
The AIE API offers a mmul class [1]_ which includes methods for performing matrix multiplication on vectorized matrices. The class is declared with the size of the multiplication (MxKxN) and a data type.
The mat function writes the result of A*B where A is MxK and B is KxN to an accumulator register. You can optionally declare a desired accumulator precision. The default precision can be found in the docs by TypeAxTypeB multiplication.

Kernel Code (kernels.cpp)
*****************************

Here is a simple example of a kernel which calculates a 4x4x4 matrix multiplication for int16.

.. code-block:: cpp

    #include <aie_api/aie.hpp>
    #include <adf.h>
    #include "kernels.h"
    #include "include.h"


    using MMUL = aie::mmul<4, 4, 4, int16, int16>; // 4x4x4 int16 config[1][3][5]

    void matmul_4x4(
        adf::input_buffer<int16>& a, adf::input_buffer<int16>& b, adf::output_buffer<int16>& c
    ) {
        // Use vector iterators for 4x4 matrices (16 elements)
        auto a_iter = aie::begin_vector<16>(a);
        auto b_iter = aie::begin_vector<16>(b);
        auto c_iter = aie::begin_vector<16>(c);

        MMUL m;
        m.mul(*a_iter, *b_iter);  // Load via iterator dereference
        
        *c_iter = m.to_vector<int16>(); // Store via iterator assignment
    }

To complete the single-tile example, we need to add the graph and host code...

Graph Code (graph.h)
********************

.. code-block:: cpp

    #include <adf.h>
    #include "kernels.h"
    #include "include.h"
    #include <aie_api/aie_adf.hpp>

    using namespace adf;


    /**
    * Single-tile matrix multiplication graph for Xilinx AI Engine using built-in mmul class
    * Optimized for int32 data type
    * A: Input matrix of dimensions 4 x 4
    * B: Input matrix of dimensions 4 x 4
    * C: Output matrix of dimensions 4 x 4
    */


    class MatrixMultInt16 : public adf::graph {
        private:
            kernel k;
        public:
            input_plio in_A;
            input_plio in_B;
            output_plio out_C;

            MatrixMultInt16() {

                in_A = input_plio::create(plio_64_bits, "data/A_matrix.txt");
                in_B = input_plio::create(plio_64_bits, "data/B_matrix.txt");
                out_C = output_plio::create(plio_64_bits, "data/C_output.txt");

                kernel k = kernel::create(matmul_4x4);

                runtime<ratio>(k) = 0.9;

                // Connect I/O ports to kernel
                dimensions(k.in[0]) = {16};
                dimensions(k.in[1]) = {16};
                dimensions(k.out[0]) = {16};

                connect(in_A.out[0], k.in[0]);
                connect(in_B.out[0], k.in[1]);
                connect(k.out[0], out_C.in[0]);

                source(k) = "src/kernels/kernels.cpp";
            }
        
    };

Host Code (graph.cpp)
*********************

.. code-block:: cpp

    #include <adf.h>
    #include "kernels.h"
    #include "graph.h"
    #include "include.h"

    using namespace adf;

    MatrixMultInt16 mmul_graph;

    int main(void) {
        mmul_graph.init();
        mmul_graph.run(1);
        mmul_graph.end();
        return 0;
    }

Simple Multi-Tile Example (16x16x16)
------------------------------------
If you need to calculate larger matrix multiplications within one tile, it is necessary to split the matrices into blocks and/or accumulate partial results. The mmul class provides the mac() function with will perform acc = acc + A*B where acc is the (vectorized) values in the accumulator register.

Note that the AIE API docs provide template kernel code for calculating larger matrices within one tile [1]_, but here we give our own simple implementation.

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

*Include here diagram of the tiling scheme.*

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

                    // Connect A row block (4x16 = 64 elements)
                    connect(a_in.out[0], mmul[row][col].in[0]);
                    dimensions(mmul[row][col].in[0]) = {256}; 

                    // Connect B column block (16x4 = 64 elements)
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
These parameters are streamed in as initial data packets prior to the other data.

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

.. rubric:: References
.. [1] AIE API mmul Class. https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__mmul.html
