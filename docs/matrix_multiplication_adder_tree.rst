Multi-Tile Matrix Multiplication with Adder-Tree
====================================================
Note that in the Simple Multi-Tile Matrix Multiplication Example (16x16x16), the entire inner dimension of the matrix multiplication was kept intact and was not split across different tiles. This inner dimension is the symbolized by K in an NxKxM matrix multplication which means AxB where A is MxK and B is KxN.

Keeping the inner dimesion intact means that each kernel to perform matrix multiplication directly without summing results between kernels. This is useful for smaller matrices, but when the inner dimension is large, in order to perserve memory and computes bounds it becomes necessary to split the inner dimension and sum results between kernels.

There are multiple ways to communicate and accumlate outputs between kernels. One method would be to use the cascade stream which streams data directly between accumulator registers of neighboring kernels. Instead, this code example places matrix multiplication tiles adjacent to a central addition kernel (``add_tree.cpp``). The addition tile reads the output buffers directly from its neighboring tile's memory and accumulates partial results.


This adder-tree method is illustrated with an int16 4x128x128 matrix multiplication example across 4 tiles. Here, the matrix multiplication kernel (``mmul_skinny.cpp``) uses aie::mac and aie::mul intrinsics for higher configurability. Note that the input K dimension must be a multiple of 32.

File structure:
::
  .
  ├── graph.hpp # Declaration of the graph
  ├── graph.cpp  # Initializes, runs, and ends the graph.
  ├── kernels
  │   └── kernels.cpp # matmul kernel implementation
  │   └── add_tree.cpp # addition kernel implementation
  └── kernels.hpp # declarations of kernel

Kernel Code
*************
Note that the additional kernel is labeled ``mmul_skinny`` because it works on dimensions 4x32x128. So, the A matrix is relatively skinny with a large K dimension.
The global variables N = 4, M = 128, K = 128 match the multiplication dimension. The K_Tile variable reflects the inner dimension of the matrix multiplication that is performed on each of the 4 tiles. The inner dimension is split evenly across the 4 tiles, so K_Tile = K/4.

Additionally, note that that multiplying 2 int16 vector registers through aie::mac fills a default 48-bit accmulator register. These results are cast back to int16 vectors with the to_vector<int16>() method. This method automatically saturates the data to avoid integer overflow.

``kernels.cpp``:
::
    #include <aie_api/aie.hpp>
    #include <aie_api/aie_adf.hpp>
    #include "include.h"
    #include "kernels.h"

    int K_Tile = K/4;

    using namespace adf;

    void mmul_skinny(
        input_buffer<int16>& a_buf,
        input_buffer<int16>& b_buf, 
        output_buffer<int16>& c_buf,
        int a_block)
    {
        auto a_iter = aie::begin_vector<32>(a_buf) + K_Tile/32*a_block;
        auto c_iter = aie::begin(c_buf);

        for (int n = 0; n < N; ++n) {
            auto b_iter = aie::begin_vector<32>(b_buf);
            for (int m = 0; m < M; ++m) {
                aie::accum<acc48, 32> acc = aie::zeros<acc48, 32>();
            aie::vector<int16, 32> a_vec;
            aie::vector<int16, 32> b_vec;
            for (int k = 0; k < K_Tile/32; ++k) {
                a_vec = *a_iter++;
                b_vec = *b_iter++;
                    
                if (k == 0) {
                    acc = aie::mul(a_vec, b_vec);
                } else {
                acc = aie::mac(acc, a_vec, b_vec);
                }
            }
            a_iter -= K_Tile/32;
                aie::vector<int16, 32> res_vec = acc.template to_vector<int16>();
                *c_iter++ = res;  
            }
            a_iter += K/32;
        }
    }

``add_tree.cpp``:
::
    #include <aie_api/aie.hpp>
    #include <aie_api/aie_adf.hpp>
    #include "include.h"
    #include "kernels.h"

    using namespace adf;

    // Assume N*M is multiple of 32 for vector alignment
    void add_tree(
        input_buffer<int16>& in0,  // North neighbor
        input_buffer<int16>& in1,  // East neighbor  
        input_buffer<int16>& in2,  // South neighbor
        input_buffer<int16>& in3,  // West neighbor
        output_buffer<int16>& out
    ) {
        // Vector iterators for 32-element parallel processing
        auto in0_iter = aie::begin_vector<32>(in0);
        auto in1_iter = aie::begin_vector<32>(in1);
        auto in2_iter = aie::begin_vector<32>(in2);
        auto in3_iter = aie::begin_vector<32>(in3);
        auto out_iter = aie::begin_vector<32>(out);

        constexpr int VEC = 32;
        const int total_vectors = (N * M) / VEC;

        // Process all elements in vector chunks
        for(int i = 0; i < total_vectors; ++i) {
            aie::vector<int16, VEC> v0 = *in0_iter++;
            aie::vector<int16, VEC> v1 = *in1_iter++;
            aie::vector<int16, VEC> v2 = *in2_iter++;
            aie::vector<int16, VEC> v3 = *in3_iter++;

            // Vector addition with saturation
            aie::vector<int16, VEC> sum = aie::add(aie::add(v0, v1), 
                                                aie::add(v2, v3));
            
            *out_iter++ = sum;
        }
    }

Graph Code
***************
The a_block_param ports are used to pass the block index which corresponds to the section of the A matrix that should be processed by each kernel. Each kernel is carefully mapped to a specific adjacent tile to ensure direct read/write buffer access between the addition tile and the multiplication tiles. Without direct buffering between adjacent tiles, bandwidth may be lowered when data is forced to be streamed through the 32 bit AXI4 interface. 

``graph.hpp``:
::
    #include <adf.h>
    #include "kernels.h"
    #include <aie_api/aie_adf.hpp>
    #include "include.h"

    using namespace adf;

    class SingleTileTest : public adf::graph {
    public:
            kernel mmul[4];
        kernel add;

        input_plio in_A;
        input_plio in_B[4];
        output_plio out_C;

        input_port a_block_param[4];

        SingleTileTest() {
            in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
            out_C = output_plio::create(plio_128_bits, "data/C_output.txt");

        add = kernel::create(add_tree);
        source(add) = "src/kernels/add_tree.cpp";
        runtime<ratio>(add) = 1.0;
        dimensions(add.out[0]) = {N*M};
            connect(add.out[0], out_C.in[0]);

        for (int i = 0; i < 4; ++i) {
            dimensions(add.in[i]) = {N*M};
                in_B[i] = input_plio::create(plio_128_bits, "data/B_"+std::to_string(i)+ ".txt");
            
                mmul[i] = kernel::create(mmul_skinny);

                runtime<ratio>(mmul[i]) = 1.0;

                dimensions(mmul[i].in[0]) = {N*K};
                dimensions(mmul[i].in[1]) = {M*(K/4)};
                dimensions(mmul[i].out[0]) = {N*M};

                connect(in_A.out[0], mmul[i].in[0]);
                connect(in_B[i].out[0], mmul[i].in[1]);
                connect(mmul[i].out[0], add.in[i]);

            // Connect parameter ports
            connect(a_block_param[i], mmul[i].in[2]);

                source(mmul[i]) = "src/kernels/kernels.cpp";
        }
        location<kernel>(add) = tile(0, 1);
        location<kernel>(mmul[0]) = tile(0, 0);
        location<kernel>(mmul[1]) = tile(1, 1);
        location<kernel>(mmul[2]) = tile(0, 2);
        location<kernel>(mmul[3]) = tile(1, 0);
        }
    };

AIE Grid View
****************
Through software simulation, the kernel layout is visualized. Note how output buffers of the mmul kernels are read directly into the adder tree kernel, bypassing the AXI4 stream.

.. image:: image/4x128x128_array.svg
   :alt: Adder-tree Matmul Grid Layout 
   :width: 600px
   :align: center
