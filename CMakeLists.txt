cmake_minimum_required(VERSION 3.28)
project(kmeans_mnist)
enable_language(CUDA)

# Rechercher les bibliothèques nécessaires
FIND_PACKAGE(CURL REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)

# Inclure les répertoires nécessaires
INCLUDE_DIRECTORIES(${CURL_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS})
SET(CMAKE_CXX_STANDARD 17)

# Spécifiez les fichiers sources
file(GLOB hdr *.hpp *.h)
file(GLOB cu *.cu)
SET(kmeans_files kmeans_cuda.cu mnist_main.cpp)

# Créer l'exécutable
ADD_EXECUTABLE(kmeans_mnist ${kmeans_files} ${hdr} ${cu})

# Lier les bibliothèques nécessaires
TARGET_LINK_LIBRARIES(kmeans_mnist ${CURL_LIBRARIES} ${ZLIB_LIBRARIES})
