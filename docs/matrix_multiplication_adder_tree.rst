Multi-Tile Matrix Multiplication with Adder-Tree
====================================================
In the Simple Multi-Tile Matrix Multiplication Example (16x16x16), the entire inner dimension (K) of the matrix multiplication is kept intact and not split across tiles. Here, K represents the inner dimension in an NxKxM multiplication, where A is MxK and B is KxN.

Keeping the inner dimension whole allows each kernel to perform matrix multiplication independently, without needing to sum results between kernels. This is efficient for small matrices, but for large K, splitting the inner dimension and accumulating results between kernels helps manage memory and compute resources.

There are several ways to combine outputs between kernels. One approach is to use the cascade stream to pass data directly between accumulator registers of neighboring kernels. In this example, matrix multiplication tiles are placed next to a central addition kernel (``add_tree.cpp``), which reads output buffers from its neighbors and accumulates the partial results.

This design is inspired by MaxEva, a framework which maximizes matrix multiplication across tiles. [1]_

.. figure:: image/adder-tree.png
   :alt: Adder-tree tiling scheme
   :width: 600px
   :align: center

This adder-tree method is demonstrated with an int16 4x128x128 matrix multiplication spread across 4 tiles. The matrix multiplication kernel (``mmul_skinny.cpp``) uses ``aie::mac`` and ``aie::mul`` intrinsics for flexibility. The K dimension must be a multiple of VEC. VEC is the vectorization factor, and VEC = 16 in our case because of length limits on the default 48-bit accumulator register for int16.

File structure:

::

  .
  ├── graph.h # Declaration of the graph
  ├── graph.cpp  # Initializes, runs, and ends the graph.
  ├── kernels
  │   └── mmul.cpp # matmul kernel implementation
  │   └── add_tree.cpp # addition kernel implementation
  └── kernels.h # declarations of kernel

Kernel Code
*************
The multiplication kernel is called ``mmul_skinny`` because it operates on 4x32x128 dimensions, where the A matrix is skinny with a large K.

    The global variables N = 4, M = 128, K = 128 set the multiplication size. The variable K_Tile represents the portion of the inner dimension handled by each of the 4 tiles, so K_Tile = K/4.

When multiplying two int16 vectors with ``aie::mac``, results are stored in a 48-bit accumulator. These are cast back to int16 vectors using ``to_vector<int16>()``, which automatically saturates the data to prevent overflow.

This approach uses kernel classes with attributes for configuration. The graph code instantiates and runs these kernel classes, passing the appropriate parameters. The ``a_block`` parameter in ``mmul_skinny`` acts like a thread ID, determining which part of the A matrix each kernel processes. The B matrix is split into 4x128 blocks, so no ``b_block`` parameter is needed. While previous designs used scalar port streaming for this, the class-based approach sets these parameters at compile time.

``kernels.h``:

::

    #include <adf.h>

    using namespace adf;

    // Matrix multiplication kernel for tiled processing
    class mmul_skinny {
        int K;       // Inner dimension (cols of A / rows of B)
        int M;       // Columns of output matrix (C)
        int T;       // Number of tiles
        int a_block; // Tile/block ID for parallel execution
    public:
        mmul_skinny(int K_val, int M_val, int T_val, int id) 
            : K(K_val), M(M_val), T(T_val), a_block(id) {}

        void run(adf::input_buffer<int16>& a_buf,
                adf::input_buffer<int16>& b_buf,
                adf::output_buffer<int16>& c_buf);

        // ADF framework registration (exposes parameters to toolchain)
        static void registerKernelClass() {
            REGISTER_FUNCTION(mmul_skinny::run);
            REGISTER_PARAMETER(K);
            REGISTER_PARAMETER(M);
            REGISTER_PARAMETER(T);
            REGISTER_PARAMETER(a_block);
        }
    };

    // 4-input adder tree for output matrix columns (M)
    class add_tree_4 {
        int M; // Columns of output matrix
    public:
        add_tree_4(int M_val) : M(M_val) {}

        void run(
            adf::input_buffer<int16>& in0, 
            adf::input_buffer<int16>& in1,  
            adf::input_buffer<int16>& in2,
            adf::input_buffer<int16>& in3,
            adf::output_buffer<int16>& out);

        static void registerKernelClass() {
            REGISTER_FUNCTION(add_tree_4::run);
            REGISTER_PARAMETER(M);
        }
    };


``mmul.cpp``:

.. code-block:: cpp

    #include <aie_api/aie.hpp>
    #include <aie_api/aie_adf.hpp>
    #include "include.h"
    #include "kernels.h"

    using namespace adf;

    void mmul_skinny::run(
        input_buffer<int16>& a_buf,
        input_buffer<int16>& b_buf, 
        output_buffer<int16>& c_buf)
    {
        // Set hardware-accelerated rounding/saturation modes
        aie::set_rounding(aie::rounding_mode::symmetric_zero);	   
        aie::set_saturation(aie::saturation_mode::saturate);

        const int K_Tile = K/T; // Tiling factor for dimension K

        // Offset A-matrix pointer for parallel block processing
        auto a_iter = aie::begin_vector<VEC>(a_buf) + K_Tile/VEC*a_block;
        auto c_iter = aie::begin(c_buf);

        // Outer loop over N - batch dimension
        for (int n = 0; n < N; ++n) {
            auto b_iter = aie::begin_vector<VEC>(b_buf);
            // M dimension - rows of output matrix
            for (int m = 0; m < M; ++m) {
                aie::accum<acc48, VEC> acc = aie::zeros<acc48, VEC>();
            aie::vector<int16, VEC> a_vec;
            aie::vector<int16, VEC> b_vec;
            
            // Inner K_Tile loop unrolled by VEC (vectorization factor)
            for (int k = 0; k < K_Tile/VEC; ++k) {
                a_vec = *a_iter++;  // Vector load from A
                b_vec = *b_iter++;  // Vector load from B
                    
                // First multiply, subsequent MACs (fused multiply-accumulate)
                acc = (k == 0) ? aie::mul(a_vec, b_vec) : aie::mac(acc, a_vec, b_vec);
            }
            a_iter -= K_Tile/VEC; // Rewind A pointer for next M iteration

                // Convert 48b accumulator to 16b with saturation
                aie::vector<int16, VEC> res_vec16 = acc.to_vector<int16>();

                // Horizontal sum of vector results
                int16 res = aie::reduce_add(res_vec16);
                *c_iter++ = res; // Store scalar result
            }
            a_iter += K/VEC; // Advance A pointer to next N block
        }
    }

Here is the kernel for the adder tree. It reads 4 input buffers from neighboring tiles and sums them together.

``add_tree.cpp``:

::

    #include <aie_api/aie.hpp>
    #include <aie_api/aie_adf.hpp>
    #include "include.h"
    #include "kernels.h"

    using namespace adf;

    // Assume N*M is multiple of VEC for vector alignment
    void add_tree_4::run(
        input_buffer<int16>& in0,  // North neighbor
        input_buffer<int16>& in1,  // East neighbor  
        input_buffer<int16>& in2,  // South neighbor
        input_buffer<int16>& in3,  // West neighbor
        output_buffer<int16>& out
    ) {
        // Vector iterators for VEC-element parallel processing
        auto in0_iter = aie::begin_vector<VEC>(in0);
        auto in1_iter = aie::begin_vector<VEC>(in1);
        auto in2_iter = aie::begin_vector<VEC>(in2);
        auto in3_iter = aie::begin_vector<VEC>(in3);
        auto out_iter = aie::begin_vector<VEC>(out);

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
 Each kernel is carefully mapped to a specific adjacent tile to ensure direct read/write buffer access between the addition tile and the multiplication tiles. Without direct buffering between adjacent tiles, bandwidth may be lowered when data is forced to be streamed through the 32 bit AXI4 interface. 

 Note the change in syntax when calling the kernels. The kernels are called with the correct values for the class wrapper parameters. K = 128, M = 128, T = 4, and a_block is set to the tile ID (0-3).

``graph.hpp``:

::

    #include <adf.h>
    #include "kernels.h"
    #include <aie_api/aie_adf.hpp>
    #include "include.h"

    using namespace adf;

    // Graph for 4-tile, 128x128 matrix multiplication
    class mmul_4x128x128 : public adf::graph {
    private:
        const unsigned int K = 128; // Inner dimension
        const unsigned int M = 128; // Output columns
        const unsigned int T = 4;   // Number of tiles

    public:
        kernel mmul[4]; // 4 parallel matrix multiplication kernels
        kernel add;     // 4-input adder tree kernel

        input_plio in_A;      // Input for matrix A
        input_plio in_B[4];   // 4 inputs for matrix B tiles
        output_plio out_C;    // Output for matrix C

        mmul_128x128() {
            // Create input and output streams
            in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
            out_C = output_plio::create(plio_128_bits, "data/C_output.txt");

            // Create 4-input adder tree kernel for final accumulation
            add = kernel::create_object<add_tree_4>(M);
            source(add) = "src/kernels/add_tree.cpp";
            runtime<ratio>(add) = 1.0;

            dimensions(add.out[0]) = {N*M};   // Output is full matrix
            dimensions(add.in[4]) = {M};      // Each input is a column block

            connect(add.out[0], out_C.in[0]); // Connect adder output to final output

            // Instantiate and connect 4 mmul kernels, each handling a tile
            for (unsigned int i = 0; i < N; ++i) {
                dimensions(add.in[i]) = {N*M};
                in_B[i] = input_plio::create(plio_128_bits, "data/B_"+std::to_string(i)+ ".txt");

                // Each mmul kernel gets its tile/block ID as 'i'
                mmul[i] = kernel::create_object<mmul_skinny>(K, M, T, i);
                runtime<ratio>(mmul[i]) = 1.0;

                // Set input/output buffer shapes for each kernel
                dimensions(mmul[i].in[0]) = {N*K};         // A: full row block
                dimensions(mmul[i].in[1]) = {M*(K/T)};     // B: tile (partitioned K)
                dimensions(mmul[i].out[0]) = {N*M};        // Output: full matrix block

                // Connect data streams
                connect(in_A.out[0], mmul[i].in[0]);       // Broadcast A to all mmuls
                connect(in_B[i].out[0], mmul[i].in[1]);    // Unique B for each tile
                connect(mmul[i].out[0], add.in[i]);        // Each mmul feeds one adder input

                source(mmul[i]) = "src/kernels/mmul.cpp";
            }

            // Map kernels to hardware tiles for parallel execution
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

.. [1] MaxEva: Maximizing the Efficiency of Matrix Multiplication on Versal AI Engine https://arxiv.org/pdf/2311.04980v2