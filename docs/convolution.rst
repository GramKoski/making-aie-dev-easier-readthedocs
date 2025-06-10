Convolution Example
======================

While, explicit convolution is not necessarily the most "natural" operation for the AI Engine. A simple example of 2D convolution with a 3x3 mask on an 8x8 image is provided for reference.

Kernel code (kernel.cpp)
--------------------------

::
    
    #include <aie_api/aie.hpp>
    #include <aie_api/aie_adf.hpp>
    #include "../kernels.h"
    #include "include.h"

    using namespace adf;

    /**
    * 2D Convolution implementation for AIE
    * Performs a 3x3 convolution on input data
    * 
    * @param input - Input buffer containing the input image data
    * @param weights - Input buffer containing the 3x3 convolution kernel weights
    * @param output - Output buffer for the convolution result
    */
    void conv2d_3x3(
        input_buffer<int16>& __restrict input,
        input_buffer<int16>& __restrict weights,
        output_buffer<int16>& __restrict output
    ) {
        // Get vector iterators for the input, weights, and output
        auto inPtr = input.data();
        auto wPtr = weights.data();
        auto outPtr = output.data();
        
        // Load the 3x3 convolution kernel weights into local memory
        int16 w[KERNEL_HEIGHT][KERNEL_WIDTH];
        for (int i = 0; i < KERNEL_HEIGHT; i++) {
            for (int j = 0; j < KERNEL_WIDTH; j++) {
                w[i][j] = wPtr[i * KERNEL_WIDTH + j];
            }
        }
        
        // Perform the convolution
        for (int y = 0; y < OUTPUT_HEIGHT; y++) {
            for (int x = 0; x < OUTPUT_WIDTH; x++) {
                int16 sum = 0;
                
                // 3x3 convolution kernel
                for (int ky = 0; ky < KERNEL_HEIGHT; ky++) {
                    for (int kx = 0; kx < KERNEL_WIDTH; kx++) {
                        int16 pixel = inPtr[(y + ky) * INPUT_WIDTH + (x + kx)];
                        sum += pixel * w[ky][kx];
                    }
                }
                
                // Write result to output
                outPtr[y * OUTPUT_WIDTH + x] = sum;
            }
        }
}

Graph code (graph.h)
----------------------

::

    #include <adf.h>
    #include "kernels.h"
    #include "kernels/include.h"
    #include <aie_api/aie_adf.hpp>

    using namespace adf;

    /**
    * Single-tile 2D convolution graph for Xilinx AI Engine
    * Optimized for int16 data type
    * Performs a 3x3 convolution on an 8x8 input image
    */
    class Convolution2DGraph : public adf::graph {
    private:
        kernel k;
    public:
        input_plio in_data;
        input_plio in_weights;
        output_plio out_result;

        Convolution2DGraph() {
            // Create input/output interfaces
            in_data = input_plio::create(plio_64_bits, "data/input_data.txt");
            in_weights = input_plio::create(plio_64_bits, "data/weights.txt");
            out_result = output_plio::create(plio_64_bits, "data/output.txt");
            
            // Create the convolution kernel
            k = kernel::create(conv2d_3x3);

            // Set runtime ratio
            runtime<ratio>(k) = 0.9;

            // Set dimensions for data ports
            dimensions(k.in[0]) = {INPUT_HEIGHT * INPUT_WIDTH};
            dimensions(k.in[1]) = {KERNEL_HEIGHT * KERNEL_WIDTH};
            dimensions(k.out[0]) = {OUTPUT_HEIGHT * OUTPUT_WIDTH};

            // Connect I/O ports to kernel
            connect(in_data.out[0], k.in[0]);
            connect(in_weights.out[0], k.in[1]);
            connect(k.out[0], out_result.in[0]);

            // Set source file for kernel
            source(k) = "kernels/kernels.cpp";
        }
    };

Host Code (host.cpp)
-----------------------

::

    #include <adf.h>
    #include "graph.h"

    using namespace adf;

    Convolution2DGraph convolution_graph;

    int main() {
        convolution_graph.init();

        // Execute the program (Start streaming and kernel execution)
        convolution_graph.run(1);
        convolution_graph.end();
        
        return 0;
    }