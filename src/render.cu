#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line) {
    cudaError_t err = cudaGetLastError();
    spdlog::error("{} ({}, line: {})", msg, fname, line);
    spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

struct rgba8_t {
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};

rgba8_t heat_lut(float x) {
    assert(0 <= x && x <= 1);

    float x0 = 1.f / 4.f;
    float x1 = 2.f / 4.f;
    float x2 = 3.f / 4.f;

    if (x < x0) {
        auto g = static_cast<std::uint8_t>(x / x0 * 255);

        return rgba8_t{0, g, 255, 255};
    } else if (x < x1) {
        auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);

        return rgba8_t{0, 255, b, 255};
    }
    else if (x < x2) {
        auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);

        return rgba8_t{r, 255, 0, 255};
    } else {
        auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);

        return rgba8_t{255, b, 0, 255};
    }
}

__device__ int clamp(int a, int a1, int a2, int min, int max)
{
    int percentage = (a - a2) / (a2 - a1);

    return percentage * (max - min) + min;
}

// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch) {
    float denum = width * width + height * height;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int n = 100;
    int mx0 = clamp(x, 0, width, -2.5, 1);
    int my0 = clamp(y, 0, height, -1, 1);

    float mx = 0.0;
    float my = 0.0;

    int i = 0;
    int mxtemp = 0;

    while (mx * mx + my * my < 2 * 2 && i < n) {
        mxtemp = mx * mx - my * my + mx0;
        my = 2 * mx * my + my0;
        mx = mxtemp;
        ++i;
    }

    uchar4* lineptr = (uchar4*)(buffer + my * pitch);
    uint8_t v = 255 * i / n;
    lineptr[mx] = {v, v, v, 255};
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations) {
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    char* devBuffer;
    size_t pitch;

    rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
    if (rc)
        abortError("Fail buffer allocation");

    // Run the kernel with blocks of size 64 x 64
    {
        int bsize = 32;
        int w = std::ceil((float)width / bsize);
        int h = std::ceil((float)height / bsize);

        spdlog::debug("running kernel of size ({},{})", w, h);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);
        mykernel <<< dimGrid, dimBlock >>> (devBuffer, width, height, pitch);

        if (cudaPeekAtLastError())
            abortError("Computation Error");
    }

    // Copy back to main memory
    rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory");
}
