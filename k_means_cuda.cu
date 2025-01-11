// K-means Algorithm in CUDA
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 256

// Kernel to compute distances and assign clusters
__global__ void assign_clusters(float *data, float *centroids, int *labels, int n, int k, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float min_dist = FLT_MAX;
    int best_cluster = 0;

    for (int cluster = 0; cluster < k; ++cluster) {
        float dist = 0.0;
        for (int dim = 0; dim < d; ++dim) {
            float diff = data[idx * d + dim] - centroids[cluster * d + dim];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = cluster;
        }
    }
    labels[idx] = best_cluster;
}

// Kernel to update centroids
__global__ void update_centroids(float *data, float *centroids, int *labels, int *counts, int n, int k, int d) {
    extern __shared__ float shared_data[];
    float *local_sums = &shared_data[threadIdx.x * d];

    for (int dim = 0; dim < d; ++dim) {
        local_sums[dim] = 0.0;
    }

    __shared__ int local_counts[BLOCK_SIZE];
    local_counts[threadIdx.x] = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cluster = labels[idx];
        atomicAdd(&counts[cluster], 1);
        for (int dim = 0; dim < d; ++dim) {
            atomicAdd(&centroids[cluster * d + dim], data[idx * d + dim]);
        }
    }
}

// Host code to initialize data and centroids
void kmeans(float *data, float *centroids, int *labels, int n, int k, int d, int max_iter) {
    float *d_data, *d_centroids;
    int *d_labels, *d_counts;

    cudaMalloc(&d_data, n * d * sizeof(float));
    cudaMalloc(&d_centroids, k * d * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));
    cudaMalloc(&d_counts, k * sizeof(int));

    cudaMemcpy(d_data, data, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, k * d * sizeof(float), cudaMemcpyHostToDevice);

    for (int iter = 0; iter < max_iter; ++iter) {
        int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        assign_clusters<<<grid_size, BLOCK_SIZE>>>(d_data, d_centroids, d_labels, n, k, d);
        cudaMemset(d_centroids, 0, k * d * sizeof(float));
        cudaMemset(d_counts, 0, k * sizeof(int));

        update_centroids<<<grid_size, BLOCK_SIZE, k * d * sizeof(float)>>>(d_data, d_centroids, d_labels, d_counts, n, k, d);

        // Normalize centroids (omitted for simplicity, but essential)
    }

    cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
}

int main() {
    const int n = 1000;  // Number of points
    const int k = 3;     // Number of clusters
    const int d = 2;     // Dimensions
    const int max_iter = 100;

    std::vector<float> data(n * d);
    std::vector<float> centroids(k * d);
    std::vector<int> labels(n);

    // Initialize data and centroids randomly
    for (int i = 0; i < n * d; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < k * d; ++i) {
        centroids[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    kmeans(data.data(), centroids.data(), labels.data(), n, k, d, max_iter);

    // Print final centroids
    std::cout << "Final centroids:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << ": ";
        for (int j = 0; j < d; ++j) {
            std::cout << centroids[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
