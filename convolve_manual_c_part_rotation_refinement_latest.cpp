#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <iostream>

struct ScanResult {
    int score;
    double angle; // in degrees
};

// Helper: Compute the convolution score and fill convResult
// Returns the maximum value (score) over the region [startX, endX) x [startY, endY).
int compute_convolution_score(
    const std::vector<double>& x_lidar_rotated,
    const std::vector<double>& y_lidar_rotated,
    const std::vector<std::vector<int>>& occupancyMap,
    std::vector<std::vector<int>>& convResult,
    int startX,
    int endX,
    int startY,
    int endY,
    const double startAngle,
    const int windowsize)
{
    // Reset the region in convResult to zero
    for (int i = startX; i < endX; ++i) {
        for (int j = startY; j < endY; ++j) {
            convResult[i][j] = 0;
        }
    }

    // Pre-compute integer lidar offsets
    std::vector<std::pair<int,int>> xyPairsScaled;
    size_t numPoints = x_lidar_rotated.size();
    for (size_t i = 0; i < numPoints; ++i) {
        int x = static_cast<int>(x_lidar_rotated[i]);
        int y = static_cast<int>(y_lidar_rotated[i]);
        xyPairsScaled.push_back(std::make_pair(x, y));
    }
    
    int bestScore = 0;
    // Loop over possible offsets
    for (int i = startX; i < endX; ++i) {
        for (int j = startY; j < endY; ++j) {
            int current = 0;
            for (const auto& pt : xyPairsScaled) {
                int x = pt.first + i;
                int y = pt.second + j;
                if (x >= 0 && x < 1600 && y >= 0 && y < 1600 && occupancyMap[x][y] == 1) {
                    current += occupancyMap[x][y];
                }
            }
            convResult[i][j] = current;
            if (current > bestScore) {
                bestScore = current;
            }
        }
    }
    return bestScore;
}

// The hierarchical search function.
// It performs a coarse search (5° increments) followed by a fine search (1° increments)
// in a ±5° window around the top 5 coarse candidate angles.
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
    // Convert raw lidar arrays into vectors.
    std::vector<double> x_lidar_local(x_lidar, x_lidar + numPoints);
    std::vector<double> y_lidar_local(y_lidar, y_lidar + numPoints);

    // Convert flattened occupancyMap into a 2D vector.
    std::vector<std::vector<int>> occupancyMap(1600, std::vector<int>(1600, 0));
    for (int i = 0; i < 1600; ++i) {
        for (int j = 0; j < 1600; ++j) {
            occupancyMap[i][j] = occupancyMap_flat[i * 1600 + j];
        }
    }
    
    // Create a full-sized convolution result matrix.
    std::vector<std::vector<int>> convResult(1600, std::vector<int>(1600, 0));
    std::vector<std::vector<int>> bestConvResult(1600, std::vector<int>(1600, 0));
    
    std::vector<ScanResult> coarseResults;
    
    // Coarse search: loop in 5° increments
    for (double angle = startAngle-windowsize; angle < startAngle+windowsize; angle += 5) {
        double rad = angle * M_PI / 180.0;
        double cosVal = cos(rad);
        double sinVal = sin(rad);
        
        // Rotate lidar points for this coarse angle.
        std::vector<double> x_rotated(numPoints);
        std::vector<double> y_rotated(numPoints);
        for (int i = 0; i < numPoints; ++i) {
            x_rotated[i] = x_lidar_local[i] * cosVal - y_lidar_local[i] * sinVal;
            y_rotated[i] = x_lidar_local[i] * sinVal + y_lidar_local[i] * cosVal;
        }
        
        // Compute convolution score for this angle.
        int currentScore = compute_convolution_score(
            x_rotated, y_rotated, occupancyMap, convResult,
            startX, endX, startY, endY, startAngle, windowsize);
        
        coarseResults.push_back({currentScore, angle});
    }
    
    // Sort coarse results by score (descending).
    std::sort(coarseResults.begin(), coarseResults.end(), [](const ScanResult &a, const ScanResult &b) {
        return a.score > b.score;
    });
    
    // Fine search: for the top 5 coarse angles, check ±5° in 1° increments.
    ScanResult bestOverall = {0, 0};
    std::vector<std::vector<int>> bestFineConvResult(1600, std::vector<int>(1600, 0));
    
    int numCandidates = std::min(3, (int)coarseResults.size());
    for (int idx = 0; idx < numCandidates; ++idx) {
        int coarseAngle = coarseResults[idx].angle;
        // Fine search range: coarseAngle ±5°
        for (int fineAngle = coarseAngle - 5; fineAngle <= coarseAngle + 5; ++fineAngle) {
            // Adjust for wrap-around.
            int adjustedAngle = (fineAngle + 360) % 360;
            double rad = adjustedAngle * M_PI / 180.0;
            double cosVal = cos(rad);
            double sinVal = sin(rad);
            
            // Rotate lidar points with fine adjustment.
            std::vector<double> x_rotated(numPoints);
            std::vector<double> y_rotated(numPoints);
            for (int i = 0; i < numPoints; ++i) {
                x_rotated[i] = x_lidar_local[i] * cosVal - y_lidar_local[i] * sinVal;
                y_rotated[i] = x_lidar_local[i] * sinVal + y_lidar_local[i] * cosVal;
            }
            
            // Compute convolution score for this fine angle.
            int currentScore = compute_convolution_score(
                x_rotated, y_rotated, occupancyMap, convResult,
                startX, endX, startY, endY, startAngle, windowsize);
            
            if (currentScore > bestOverall.score) {
                bestOverall.score = currentScore;
                bestOverall.angle = adjustedAngle;
                bestFineConvResult = convResult; // Save the best convolution result.
            }
        }
    }
    
    // Set output sum.
    *sum = bestOverall.score;
    *bestAngle = bestOverall.angle;
    
    // Copy the best convolution result back to the flattened result array.
    // (Only the region [startX, endX) x [startY, endY) is copied.)
    for (int i = startX; i < endX; ++i) {
        for (int j = startY; j < endY; ++j) {
            result[i * 1600 + j] = bestFineConvResult[i][j];
        }
    }
    
    // Optionally, you might print the best angle found:
    //std::cout << "Best angle (degrees): " << bestOverall.angle << std::endl;
}
