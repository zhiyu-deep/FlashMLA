cmake_minimum_required(VERSION 3.18)
project(FlashMLA CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_library(flash_mla csrc/flash_api.cpp csrc/flash_fwd_mla_bf16_sm90.cu)
target_include_directories(flash_mla PRIVATE csrc csrc/cutlass/include
        /home/jiangzhiyu/.local/lib/python3.8/site-packages/torch/include)