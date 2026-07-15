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
float quantizeFP8M4E3(float value) { //تابع یه float می‌گیره و یه float برمی‌گردونه


    if (!std::isfinite(value) || value == 0.0f) return value; //اگه عدد صفر یا بی‌نهایت بود، دست نزن و همونو برگردون



    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits)); //عدد float رو به بیت‌های خام تبدیل کن تا بتونیم دستکاریش کنیم



    int exp = ((bits >> 23) & 0xFF) - 127;
    exp = std::max(-6, std::min(7, exp)); //exponent رو بخون و به بازه FP8 محدود کن (فقط -6 تا +7 مجازه)


    bits = (bits & 0x80000000u)
         | (uint32_t(exp + 127) << 23)
         | (bits & 0x7FFFFF00u);
//عدد رو با exponent جدید بازسازی کن


    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return quantizeMantissa(out, 3); //mantissa رو هم به 3 بیت کم کن


}


float relError(float approx, float reference) {
    if (reference == 0.0f)
        return std::fabs(approx - reference); // absolute error
    return std::fabs((approx - reference) / reference);
}


int main() {
    // ======================== نتایج مهم ========================
    // b=0.75  (جواب صحیح=1.333): iter3: float=0.023خطای کم, bf16=0.021خطای کم, fp8=0.062 → fp8 خطای بیشتر
    // b=1.5   (جواب صحیح=0.666): fp8 از iter2 گیر می‌کنه (0.6875) و همگرا نمیشه!
    // b=3.7   (جواب صحیح=0.270): هیچکدام همگرا نمیشن چون x0=0.5 نقطه شروع خوبی نیست
    // نتیجه: float و bf16 همیشه همگرا میشن، fp8 در بعضی موارد گیر می‌کنه
   //برای b=3.7 هیچکدام همگرا نشدن - این به خاطر precision نیست، به خاطر نقطه شروع بده (x0=0.5 برای b=3.7 مناسب نیست).

//پس نتیجه دقیق‌تر اینه:

//float: با نقطه شروع مناسب، همیشه همگرا میشه ✓
//BF16: تقریباً مثل float ✓
//FP8: حتی با نقطه شروع مناسب هم ممکنه گیر کنه ✗
//chera hichkodom dar precision hamgra nashodan ? x0 باید در این بازه باشه:


//0 < x0 < 2/b
// برای b=3.7:
// 2/b = 2/3.7 = 0.54
// یعنی x0 باید بین 0 و 0.54 باشه.

//x0 = 0.5 هست که خیلی نزدیک به مرز 0.54 هست و الگوریتم کند همگرا میشه، نه اینکه precision مشکل داشته باشه.
    // ===========================================================
    // ===========================================================
     
    // Der NR-Algorithmus mit x0=0.5 konvergiert nur fuer 0 < b < 4
// Fuer b ausserhalb dieses Bereichs divergiert der Algorithmus
// الگوریتم NR با x0=0.5 فقط برای بازه 0 < b < 4 همگرا میشه



const float values[] = {
    1000.0f, 5000.0f,      // بزرگ مثبت
    0.75f, 3.7f,           // نرمال مثبت
    0.0001f, 0.00001f,     // کوچک مثبت
    -0.0001f, -0.00001f,   // کوچک منفی
    -1.5f, -3.7f,          // نرمال منفی
    -1000.0f, -5000.0f,    // بزرگ منفی
    1e-38f, 1e38f          // overflow و underflow
};

    std::cout << std::fixed << std::setprecision(8);

    for (float b : values) {
        const float reference = 1.0f / b;  // جواب صحیح = 1/b
        float xFloat = 0.5f;   // x0 = مقدار اولیه برای float
        float sign = 1.0f;
    if (b < 0.0f)
        sign = -1.0f;
       // علامت b رو جدا کن
        float absB = std::fabs(b);                // |b| مثبت

        float xBf16  = 0.5f;   // x0 = مقدار اولیه برای BF16
        float xFp8   = 0.5f;   // x0 = مقدار اولیه برای FP8 - باید در بازه (0, 2/b) باشه

        std::cout << "b = " << b << "\n";
        std::cout << "  reference 1/b = " << reference << "\n";
        std::cout << "  start x0      = 0.50000000\n";

        for (int iteration = 1; iteration <= 3; ++iteration) {
        xFloat = stepFloat(absB, xFloat);
        xBf16  = stepQuantized(absB, xBf16, 7);
        xFp8   = quantizeFP8M4E3(stepFloat(absB, xFp8));




            std::cout << "  iter " << iteration << ":\n";
           float resFloat = sign * xFloat;
        float resBf16  = sign * xBf16;
        float resFp8   = sign * xFp8;

        std::cout << "    float    = " << resFloat
          << "   rel.err = " << relError(resFloat, reference) << "\n";
        std::cout << "    bf16-ish = " << resBf16
          << "   rel.err = " << relError(resBf16, reference) << "\n";
        std::cout << "    fp8-ish  = " << resFp8
          << "   rel.err = " << relError(resFp8, reference) << "\n";

        }
        // روش جدید: b رو به بازه [1,2) ببر
int k = 0;
float bNorm = absB;
while (bNorm >= 2.0f) { bNorm /= 2.0f; k++; }  // اگه b بزرگه، نصف کن
while (bNorm < 1.0f)  { bNorm *= 2.0f; k--; }   // اگه b کوچیکه، دو برابر کن

// حالا NR روی bNorm اجرا کن - x0=0.5 همیشه کار میکنه چون bNorm بین 1 و 2 هست
float xNorm = 0.5f;
for (int i = 0; i < 3; i++)
    xNorm = stepFloat(bNorm, xNorm);

// نتیجه رو تصحیح کن: 1/b = (1/bNorm) / 2^k
float resultNorm = xNorm;
for (int i = 0; i < k; i++)  resultNorm /= 2.0f;
for (int i = 0; i > k; i--)  resultNorm *= 2.0f;

std::cout << "  normalized NR = " << sign * resultNorm
          << "   rel.err = " << relError(sign * resultNorm, reference) << "\n";

        std::cout << '\n';
    }

    return 0;
}