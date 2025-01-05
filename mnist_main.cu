#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "kmeans_cuda.cuh"  // Include the CUDA implementation header

// File paths
const char *MNIST_IMAGES_FILE = "/array/shared/home/kmessan/k_means_CUDA/train-images.idx3-ubyte";
const char *MNIST_LABELS_FILE = "/array/shared/home/kmessan/train-labels-idx1-ubyte";

// Read MNIST images
void load_mnist_images(const char *file_path, std::vector<float> &images, int &num_samples, int &num_features) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char *>(&magic_number), 4);
    file.read(reinterpret_cast<char *>(&num_images), 4);
    file.read(reinterpret_cast<char *>(&rows), 4);
    file.read(reinterpret_cast<char *>(&cols), 4);

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    num_samples = num_images;
    num_features = rows * cols;

    images.resize(num_samples * num_features);
    for (int i = 0; i < num_samples * num_features; ++i) {
        unsigned char pixel = 0;
        file.read(reinterpret_cast<char *>(&pixel), 1);
        images[i] = pixel / 255.0f;  // Normalize to [0, 1]
    }
}

int main() {
    const int k = 10;  // Number of clusters
    const int max_iter = 100;

    std::vector<float> data;
    int num_samples, num_features;

    // Load MNIST images
    load_mnist_images(MNIST_IMAGES_FILE, data, num_samples, num_features);

    std::vector<float> centroids(k * num_features);
    std::vector<int> labels(num_samples);

    // Initialize centroids randomly
    for (int i = 0; i < k * num_features; ++i) {
        centroids[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Call the kmeans function
    kmeans(data.data(), centroids.data(), labels.data(), num_samples, k, num_features, max_iter);

    // Output the results
    std::cout << "Final centroids:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << ": ";
        for (int j = 0; j < num_features; ++j) {
            std::cout << centroids[i * num_features + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}