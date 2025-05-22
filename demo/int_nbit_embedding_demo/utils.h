#pragma once

#include <cstdint>
#include <vector>
#include <cuda_fp16.h> // For CUDA half2

// Matches fbgemm_gpu::SparseType in fbgemm_gpu/src/split_embeddings_cache/common.h
// and fbgemm_gpu/fbgemm_gpu/enums.py
enum class SparseType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
    // Per fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py,
    // these are also valid.
    BF16 = 5,
    FP8 = 6, // Matches SparseType::FP8 from host_template
    // Add others if necessary from enums.py, but these cover n-bit and float types
};

// Simple struct for CPU-side half2 representation if needed,
// though direct manipulation of bytes for dev_weights might be more common.
// CUDA's half2 is for device code, but can be used for size/layout reference.
struct CPUHalf2 {
    uint16_t x;
    uint16_t y;

    // Default constructor
    CPUHalf2() : x(0), y(0) {}

    // Constructor from two float values (basic conversion, not IEEE standard)
    // A proper conversion library or ATen's methods should be used for accuracy.
    // For the demo, if we populate scales/biases, we'll likely use ATen's half.
    CPUHalf2(float val_x, float val_y) {
        // This is a placeholder. Proper float to half conversion is complex.
        // For the demo, we'll likely prepare 'half' values using PyTorch/ATen
        // on the CPU and then copy their byte representation.
        x = static_cast<uint16_t>(0); // Placeholder
        y = static_cast<uint16_t>(0); // Placeholder
    }
};

// Placeholder for PlacementType if needed, mirroring fbgemm_gpu::PlacementType
// For the demo, we'll likely just use integer constants if required by any
// utility function, or assume DEVICE placement.
enum class PlacementType : int32_t {
    DEVICE = 0,
    HOST = 1,
    MANAGED = 2,
    MANAGED_CACHING = 3,
};

// New code starts here
#include <algorithm> // For std::min/max
#include <cmath>     // For std::round, std::ldexp, std::fmax, std::fmin
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For debug prints, can be removed later
#include <cstring>   // For std::memcpy

// Function to convert a float to a half precision float (uint16_t)
// This is a simplified version. For robust conversion, a library like ATen's should be used.
// For the demo, we primarily care about the layout and bitwise operations.
inline uint16_t float_to_half_bits(float f) {
    // This is not a correct conversion, merely a placeholder for bit representation.
    // In a real scenario, use something like at::convert<at::Half, float>(f).to<uint16_t>();
    // For the demo, we'll assume scales/biases are prepared correctly if coming from PyTorch.
    // If we generate them purely in C++, this would need a proper implementation.
    // For now, let's return a predictable bit pattern if f is 0, else a non-zero pattern.
    // if (f == 0.0f) return 0;
    // A more realistic stub might involve bit-shifting a float, but it's complex.
    // Returning a fixed non-zero pattern for non-zero inputs for placeholder purposes.
    // unsigned int float_bits;
    // std::memcpy(&float_bits, &f, sizeof(float));
    // Simplified: take top 16 bits (sign, exponent, part of mantissa)
    // This is NOT a valid float-to-half conversion.
    // return (float_bits >> 16);

    // Using a slightly more involved placeholder that somewhat mimics half structure
    // to ensure we are writing meaningful data for scale/bias.
    // This is still not a substitute for a proper conversion.
    if (std::isnan(f)) return 0x7e00; // qNaN
    if (std::isinf(f)) return (f < 0) ? 0xfc00 : 0x7c00;
    if (f == 0.0f) return (std::signbit(f)) ? 0x8000 : 0x0000;

    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint16_t sign = (bits >> 16) & 0x8000;
    int exponent = ((bits >> 23) & 0xff) - 127;
    uint32_t mantissa = bits & 0x7fffff;

    if (exponent > 15) { // Overflow
        return sign | 0x7c00;
    }
    if (exponent < -14) { // Subnormal or zero
        // For simplicity, flush to zero or handle as subnormal
        mantissa = mantissa | 0x800000; // Make it normal
        int shift = -14 - exponent;
        if (shift > 24) { // Too small, flush to zero
             return sign;
        }
        mantissa >>= shift;
        return sign | (mantissa >> 13); // Mantissa for half is 10 bits
    }
    return sign | ((exponent + 15) << 10) | (mantissa >> 13);
}

// Function to convert half precision float (uint16_t) to float
// Simplified version.
inline float half_bits_to_float(uint16_t h) {
    // Placeholder - a proper conversion is needed for actual verification.
    // For the demo, if we check values, this needs to be accurate.
    // if (h == 0) return 0.0f;
    // return 1.0f; // Dummy non-zero

    uint16_t sign_val = h >> 15;
    uint16_t exponent_half = (h >> 10) & 0x1F;
    uint16_t mantissa_half = h & 0x3FF;

    if (exponent_half == 0x1F) { // Inf or NaN
        return (mantissa_half == 0) ? 
               (sign_val ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity()) :
               std::numeric_limits<float>::quiet_NaN();
    }
    if (exponent_half == 0 && mantissa_half == 0) { // Zero
        return sign_val ? -0.0f : 0.0f;
    }

    int exponent_float = exponent_half - 15 + 127; // Adjust bias
    uint32_t mantissa_float_val;

    if (exponent_half == 0) { // Denormalized
        exponent_float = 1 - 15 + 127; // Exponent for denormals
        mantissa_float_val = mantissa_half;
        // Normalize
        while (!(mantissa_float_val & 0x400)) {
            mantissa_float_val <<= 1;
            exponent_float--;
        }
        mantissa_float_val &= 0x3FF; // Remove implicit leading 1
    } else { // Normalized
        mantissa_float_val = mantissa_half;
    }

    uint32_t float_bits_val = (static_cast<uint32_t>(sign_val) << 31) |
                          (static_cast<uint32_t>(exponent_float) << 23) |
                          (static_cast<uint32_t>(mantissa_float_val) << 13);
    float result;
    std::memcpy(&result, &float_bits_val, sizeof(float));
    return result;
}


// Quantizes a row of floats to N-bit integers and prepends scale and bias.
// Output: [scale (half), bias (half), N-bit data packed into uint8_t...]
inline std::vector<uint8_t> quantize_row_nbit(const float* data, int D, int N) {
    if (N != 2 && N != 4 && N != 8) {
        throw std::runtime_error("Unsupported bit-width N. Only 2, 4, 8 are supported.");
    }
    if (D <= 0) {
        throw std::runtime_error("Dimension D must be positive.");
    }

    float min_val = data[0];
    float max_val = data[0];
    for (int i = 1; i < D; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    float scale = (max_val == min_val) ? 1.0f : (max_val - min_val) / (static_cast<float>((1 << N) - 1));
    float bias = min_val; // This is the effective 'zero_point' scaled by `scale` if we consider it as standard quantization `val = scale * q_val + bias`

    std::vector<uint8_t> quantized_row;
    quantized_row.resize(sizeof(uint16_t) * 2); // Space for scale and bias (as half floats)

    uint16_t scale_half_bits = float_to_half_bits(scale);
    uint16_t bias_half_bits = float_to_half_bits(bias);

    std::memcpy(quantized_row.data(), &scale_half_bits, sizeof(uint16_t));
    std::memcpy(quantized_row.data() + sizeof(uint16_t), &bias_half_bits, sizeof(uint16_t));
    
    // std::cout << "Quantizing Row: D=" << D << ", N=" << N << std::endl;
    // std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;
    // std::cout << "  Scale (float): " << scale << ", Bias (float): " << bias << std::endl;
    // std::cout << "  Scale (half_bits): " << scale_half_bits << ", Bias (half_bits): " << bias_half_bits << std::endl;


    int num_packed_bytes = (D * N + 7) / 8; // Ceiling division
    size_t initial_size = quantized_row.size();
    quantized_row.resize(initial_size + num_packed_bytes);
    std::fill(quantized_row.begin() + initial_size, quantized_row.end(), 0);


    int current_byte_idx = initial_size;
    int bit_offset = 0;

    for (int i = 0; i < D; ++i) {
        float normalized_val = (data[i] - bias) / (scale == 0.0f ? 1.0f : scale) ;
        uint8_t quantized_val = static_cast<uint8_t>(std::round(std::fmax(0.0f, std::fmin((float)((1 << N) - 1), normalized_val))));

        // std::cout << "  Idx " << i << ": float " << data[i] << " -> norm " << normalized_val << " -> q_val " << (int)quantized_val << std::endl;

        for (int bit = 0; bit < N; ++bit) {
            if ((quantized_val >> bit) & 1) {
                quantized_row[current_byte_idx] |= (1 << bit_offset);
            }
            bit_offset++;
            if (bit_offset == 8) {
                bit_offset = 0;
                current_byte_idx++;
            }
        }
    }
    // std::cout << "  Packed data (size " << num_packed_bytes << "): ";
    // for(int k=0; k<num_packed_bytes; ++k) {
    //    std::cout << std::hex << (int)quantized_row[initial_size + k] << " ";
    // }
    // std::cout << std::dec << std::endl;

    return quantized_row;
}

// Dequantizes a row of N-bit integers (preceded by scale and bias) back to floats.
inline std::vector<float> dequantize_row_nbit(const uint8_t* quantized_data_with_header, int D, int N) {
    if (N != 2 && N != 4 && N != 8) {
        throw std::runtime_error("Unsupported bit-width N. Only 2, 4, 8 are supported for dequantization.");
    }
     if (D <= 0) {
        throw std::runtime_error("Dimension D must be positive.");
    }

    uint16_t scale_half_bits;
    uint16_t bias_half_bits;
    std::memcpy(&scale_half_bits, quantized_data_with_header, sizeof(uint16_t));
    std::memcpy(&bias_half_bits, quantized_data_with_header + sizeof(uint16_t), sizeof(uint16_t));

    float scale = half_bits_to_float(scale_half_bits);
    float bias = half_bits_to_float(bias_half_bits);
    
    // std::cout << "Dequantizing Row: D=" << D << ", N=" << N << std::endl;
    // std::cout << "  Scale (half_bits): " << scale_half_bits << ", Bias (half_bits): " << bias_half_bits << std::endl;
    // std::cout << "  Scale (float): " << scale << ", Bias (float): " << bias << std::endl;


    std::vector<float> dequantized_row;
    dequantized_row.resize(D);

    const uint8_t* nbit_data_ptr = quantized_data_with_header + sizeof(uint16_t) * 2;
    int current_byte_idx = 0;
    int bit_offset = 0;
    // uint8_t n_bit_mask = (1 << N) - 1; // Not actually needed with current loop structure

    for (int i = 0; i < D; ++i) {
        uint8_t packed_val = 0;
        for (int bit = 0; bit < N; ++bit) {
            if ((nbit_data_ptr[current_byte_idx] >> bit_offset) & 1) {
                packed_val |= (1 << bit);
            }
            bit_offset++;
            if (bit_offset == 8) {
                bit_offset = 0;
                current_byte_idx++;
            }
        }
        dequantized_row[i] = static_cast<float>(packed_val) * scale + bias;
        // std::cout << "  Idx " << i << ": q_val " << (int)packed_val << " -> float " << dequantized_row[i] << std::endl;
    }

    return dequantized_row;
}
