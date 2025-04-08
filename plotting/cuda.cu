#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// Define a structure for candidate scan results.
struct ScanResult {
    int score;
    double angle; // in degrees
};

// -----------------------------------------------------------------------------
// CUDA Kernel: Rotate the lidar points
// Each thread rotates one point from the input arrays.
__global__ void rotationKernel(
    const double* x, const double* y, 
    double cosVal, double sinVal, 
    double* x_rot, double* y_rot, 
    int numPoints)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < numPoints) {
        x_rot[idx] = x[idx] * cosVal - y[idx] * sinVal;
        y_rot[idx] = x[idx] * sinVal + y[idx] * cosVal;
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Convolve over the region for one rotated lidar scan.
// Each thread processes one candidate translation offset (tx,ty) in the region.
__global__ void convolutionKernel(
    const double* x_rot, const double* y_rot, 
    int numPoints,
    const int* occupancyMap, int mapWidth, int mapHeight,
    int startX, int startY,
    int regionWidth, int regionHeight,
    int* convResult)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // index along region width
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // index along region height

    if(tx < regionWidth && ty < regionHeight) {
        // Convert the region-relative index to the global map coordinate.
        int globalX = startX + tx;
        int globalY = startY + ty;
        int sum = 0;
        // Loop over all lidar points
        for (int k = 0; k < numPoints; k++) {
            // For each rotated point, compute its position in the occupancy map,
            // shifted by the translation (globalX, globalY)
            int x = static_cast<int>(x_rot[k]) + globalX;
            int y = static_cast<int>(y_rot[k]) + globalY;
            // Check boundaries
            if (x >= 0 && x < mapWidth && y >= 0 && y < mapHeight) {
                // Only if the occupancy value is 1 is it counted.
                int occ = occupancyMap[x * mapWidth + y];
                if(occ == 1)
                    sum += occ;
            }
        }
        // Save the result for this offset (store row-major order).
        convResult[ty * regionWidth + tx] = sum;
    }
}

// -----------------------------------------------------------------------------
// The externally visible function remains with the same signature.
// It performs a hierarchical two-stage search (coarse then fine) using CUDA.
extern "C" void convolve_lidar_scan_c_coarse_fine(
    const double* x_lidar, const double* y_lidar, int numPoints,
    const int* occupancyMap_flat, 
    const int startX,
    const int endX,
    const int startY,
    const int endY,
    const double startAngle,
    const int windowsize,
    int* result, int* sum, int* bestAngle)
{
    // Dimensions for the occupancy map.
    const int mapWidth = 1600;
    const int mapHeight = 1600;
    // Define the convolution search region dimensions.
    int regionWidth = endX - startX;
    int regionHeight = endY - startY;

    // -------------------------------------------------------------------------
    // Copy the occupancy map (flattened) to device memory.
    size_t occupancySize = mapWidth * mapHeight * sizeof(int);
    int* d_occupancyMap = nullptr;
    cudaMalloc((void**)&d_occupancyMap, occupancySize);
    cudaMemcpy(d_occupancyMap, occupancyMap_flat, occupancySize, cudaMemcpyHostToDevice);

    // Copy lidar input arrays to device memory.
    size_t pointsSize = numPoints * sizeof(double);
    double *d_x_lidar = nullptr, *d_y_lidar = nullptr;
    cudaMalloc((void**)&d_x_lidar, pointsSize);
    cudaMalloc((void**)&d_y_lidar, pointsSize);
    cudaMemcpy(d_x_lidar, x_lidar, pointsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_lidar, y_lidar, pointsSize, cudaMemcpyHostToDevice);

    // Variables to store the coarse search results.
    std::vector<ScanResult> coarseResults;
    // Host vector to hold convolution results for the searched region.
    std::vector<int> h_convResult(regionWidth * regionHeight, 0);

    // Configure kernel launch parameters for rotation.
    int threadsPerBlock = 256;
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    // Configure kernel launch parameters for convolution.
    dim3 blockDim(16, 16);
    dim3 gridDim((regionWidth + blockDim.x - 1) / blockDim.x, (regionHeight + blockDim.y - 1) / blockDim.y);

    // -----------------------------
    // Coarse Search: check angles in steps of 5°.
    for (double angle = startAngle - windowsize; angle < startAngle + windowsize; angle += 5.0) {
        double rad = angle * M_PI / 180.0;
        double cosVal = cos(rad);
        double sinVal = sin(rad);

        // Allocate memory for rotated points on device.
        double* d_x_rot = nullptr;
        double* d_y_rot = nullptr;
        cudaMalloc((void**)&d_x_rot, pointsSize);
        cudaMalloc((void**)&d_y_rot, pointsSize);

        // Launch the rotation kernel.
        rotationKernel<<<numBlocks, threadsPerBlock>>>(d_x_lidar, d_y_lidar, cosVal, sinVal, d_x_rot, d_y_rot, numPoints);
        cudaDeviceSynchronize();

        // Allocate memory for the convolution result (region only) on device.
        int* d_convResult = nullptr;
        cudaMalloc((void**)&d_convResult, regionWidth * regionHeight * sizeof(int));

        // Launch the convolution kernel.
        convolutionKernel<<<gridDim, blockDim>>>(d_x_rot, d_y_rot, numPoints,
                                                   d_occupancyMap, mapWidth, mapHeight,
                                                   startX, startY, regionWidth, regionHeight,
                                                   d_convResult);
        cudaDeviceSynchronize();

        // Copy the convolution result back to host.
        cudaMemcpy(h_convResult.data(), d_convResult, regionWidth * regionHeight * sizeof(int), cudaMemcpyDeviceToHost);

        // Compute the maximum score in the convolution region.
        int currentScore = 0;
        for (int idx = 0; idx < regionWidth * regionHeight; idx++) {
            if (h_convResult[idx] > currentScore)
                currentScore = h_convResult[idx];
        }
        // Save the result for this angle.
        coarseResults.push_back({currentScore, angle});

        // Free temporary device memory.
        cudaFree(d_x_rot);
        cudaFree(d_y_rot);
        cudaFree(d_convResult);
    }

    // Sort the coarse results by descending score.
    std::sort(coarseResults.begin(), coarseResults.end(), [](const ScanResult &a, const ScanResult &b) {
        return a.score > b.score;
    });

    // -----------------------------
    // Fine Search: For the top coarse candidate angles (up to 3),
    // perform a finer search (±5° in 1° increments).
    ScanResult bestOverall = {0, 0.0};
    std::vector<int> bestFineConvResult(regionWidth * regionHeight, 0);

    int numCandidates = std::min(3, (int)coarseResults.size());
    for (int idx = 0; idx < numCandidates; ++idx) {
        double coarseAngle = coarseResults[idx].angle;
        // Fine search range: coarseAngle ±5°
        for (int fineAngle = static_cast<int>(coarseAngle) - 5; fineAngle <= static_cast<int>(coarseAngle) + 5; fineAngle++) {
            // Adjust for wrap-around.
            int adjustedAngle = (fineAngle + 360) % 360;
            double rad = adjustedAngle * M_PI / 180.0;
            double cosVal = cos(rad);
            double sinVal = sin(rad);

            // Allocate device memory for rotated arrays.
            double* d_x_rot = nullptr;
            double* d_y_rot = nullptr;
            cudaMalloc((void**)&d_x_rot, pointsSize);
            cudaMalloc((void**)&d_y_rot, pointsSize);

            // Launch the rotation kernel for the fine angle.
            rotationKernel<<<numBlocks, threadsPerBlock>>>(d_x_lidar, d_y_lidar, cosVal, sinVal, d_x_rot, d_y_rot, numPoints);
            cudaDeviceSynchronize();

            // Allocate convolution result device array.
            int* d_convResult = nullptr;
            cudaMalloc((void**)&d_convResult, regionWidth * regionHeight * sizeof(int));

            // Launch the convolution kernel.
            convolutionKernel<<<gridDim, blockDim>>>(d_x_rot, d_y_rot, numPoints,
                                                       d_occupancyMap, mapWidth, mapHeight,
                                                       startX, startY, regionWidth, regionHeight,
                                                       d_convResult);
            cudaDeviceSynchronize();

            // Copy the result back to host.
            cudaMemcpy(h_convResult.data(), d_convResult, regionWidth * regionHeight * sizeof(int), cudaMemcpyDeviceToHost);

            // Find the maximum score in this convolution result.
            int currentScore = 0;
            for (int idxConv = 0; idxConv < regionWidth * regionHeight; idxConv++) {
                if (h_convResult[idxConv] > currentScore)
                    currentScore = h_convResult[idxConv];
            }
            // If this fine angle improves the overall best score, update.
            if (currentScore > bestOverall.score) {
                bestOverall.score = currentScore;
                bestOverall.angle = adjustedAngle;
                bestFineConvResult = h_convResult;  // Save the best convolution result region.
            }

            // Free temporary device memory for this fine angle.
            cudaFree(d_x_rot);
            cudaFree(d_y_rot);
            cudaFree(d_convResult);
        }
    }

    // Set the output variables.
    *sum = bestOverall.score;
    *bestAngle = static_cast<int>(bestOverall.angle);

    // Copy back the best convolution result region into the flattened output array.
    // Only the region [startX, endX) x [startY, endY) is updated.
    for (int i = 0; i < regionHeight; i++) {
        for (int j = 0; j < regionWidth; j++) {
            result[(startY + i) + (startX + j) * mapHeight] = bestFineConvResult[i * regionWidth + j];

            //result[(startX + j) + (startY + i)*mapWidth] = bestFineConvResult[i * regionWidth + j];
        }
    }

    // Free allocated device memory.
    cudaFree(d_occupancyMap);
    cudaFree(d_x_lidar);
    cudaFree(d_y_lidar);
}
