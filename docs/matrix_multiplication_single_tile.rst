Single-Tile Matrix Multiplication
===========================================
The AIE API is a C++ library which provides a convenient set of functions and classes for programming kernels. We will discuss the use of this API in order to perform matrix multiplication.

The AIE API offers a mmul class [1]_ which includes methods for performing matrix multiplication on vectorized matrices. The class is declared with the size of the multiplication (MxKxN) and a data type.
The ``mmul::mat()`` function writes the result of A*B where A is MxK and B is KxN to an accumulator register. You can optionally declare a desired accumulator precision. The default precision can be found in the docs by TypeAxTypeB multiplication.

File structure:
::

  .
  ├── graph.hpp # Used for the graph
  ├── graph.cpp  # Initializes, runs, and ends the graph. 
  ├── kernels
  │   └── kernels.cpp # matmul kernel implementation
  └── kernels.hpp # declarations of kernel

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


.. rubric:: References
.. [1] AIE API mmul Class. https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__mmul.html
