#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <zlib.h>  // For decompression
#include <curl/curl.h>  // For downloading files
#include "kmeans_cuda_implementation.cuh"  // Include the CUDA implementation header

// URLs for MNIST dataset
const char *MNIST_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
const char *MNIST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";

// File paths
const char *MNIST_IMAGES_FILE = "train-images-idx3-ubyte";
const char *MNIST_LABELS_FILE = "train-labels-idx1-ubyte";

// Callback function for libcurl
size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Download a file using libcurl
void download_file(const char *url, const char *output_file) {
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize libcurl." << std::endl;
        exit(EXIT_FAILURE);
    }

    FILE *fp = fopen(output_file, "wb");
    if (!fp) {
        std::cerr << "Failed to open file for writing: " << output_file << std::endl;
        exit(EXIT_FAILURE);
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Error downloading file: " << curl_easy_strerror(res) << std::endl;
        exit(EXIT_FAILURE);
    }

    fclose(fp);
    curl_easy_cleanup(curl);
}

// Decompress a gzip file
void decompress_gzip(const char *input_file, const char *output_file) {
    gzFile gz = gzopen(input_file, "rb");
    if (!gz) {
        std::cerr << "Failed to open gzip file: " << input_file << std::endl;
        exit(EXIT_FAILURE);
    }

    FILE *out = fopen(output_file, "wb");
    if (!out) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        exit(EXIT_FAILURE);
    }

    char buffer[4096];
    int num_read;
    while ((num_read = gzread(gz, buffer, sizeof(buffer))) > 0) {
        fwrite(buffer, 1, num_read, out);
    }

    gzclose(gz);
    fclose(out);
}

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

    // Download and decompress MNIST files
    download_file(MNIST_IMAGES_URL, "train-images-idx3-ubyte.gz");
    decompress_gzip("train-images-idx3-ubyte.gz", MNIST_IMAGES_FILE);

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
