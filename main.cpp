#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

float quantizeMantissa(float value, int keptBits) {
    if (!std::isfinite(value) || value == 0.0f || keptBits >= 23) {
        return value;
    }

    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t keepMask = ~((1u << (23 - keptBits)) - 1u);
    bits = (bits + (1u << (22 - keptBits))) & keepMask;

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float stepFloat(float b, float x) {
    return x * (2.0f - b * x);
}

float stepQuantized(float b, float x, int keptBits) {
    const float qb = quantizeMantissa(b, keptBits);
    const float qx = quantizeMantissa(x, keptBits);
    const float product = quantizeMantissa(qb * qx, keptBits);
    const float correction = quantizeMantissa(2.0f - product, keptBits);
    return quantizeMantissa(qx * correction, keptBits);
}

float relError(float approx, float reference) {
    return std::fabs((approx - reference) / reference);
}

int main() {
    const float values[] = {0.75f, 1.0f, 1.5f, 3.7f};

    std::cout << std::fixed << std::setprecision(8);

    for (float b : values) {
        const float reference = 1.0f / b;
        float xFloat = 0.5f;
        float xBf16 = 0.5f;
        float xFp8 = 0.5f;

        std::cout << "b = " << b << "\n";
        std::cout << "  reference 1/b = " << reference << "\n";
        std::cout << "  start x0      = 0.50000000\n";

        for (int iteration = 1; iteration <= 3; ++iteration) {
            xFloat = stepFloat(b, xFloat);
            xBf16 = stepQuantized(b, xBf16, 7);
            xFp8 = stepQuantized(b, xFp8, 3);

            std::cout << "  iter " << iteration << ":\n";
            std::cout << "    float    = " << xFloat
                      << "   rel.err = " << relError(xFloat, reference) << "\n";
            std::cout << "    bf16-ish = " << xBf16
                      << "   rel.err = " << relError(xBf16, reference) << "\n";
            std::cout << "    fp8-ish  = " << xFp8
                      << "   rel.err = " << relError(xFp8, reference) << "\n";
        }

        std::cout << '\n';
    }

    return 0;
}