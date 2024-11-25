cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(torchlpc)

find_package(Torch REQUIRED)

# Define our library target
add_library(torchlpc SHARED torchlpc.cpp)
# Enable C++17
target_compile_features(torchlpc PRIVATE cxx_std_17)
# Link against LibTorch
target_link_libraries(torchlpc "${TORCH_LIBRARIES}")
