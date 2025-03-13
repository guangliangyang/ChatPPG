import math

def quantize(x, precision=10**-3):
    """
    将浮点数 x 按给定精度量化为整数。
    精度为 10^-3 时，即乘以 1000 后取整。
    """
    factor = int(1/precision)  # 当 precision=0.001, factor=1000
    return int(round(x * factor))

def dequantize(x_int, precision=10**-3):
    """
    将量化后的整数恢复为浮点数。
    """
    factor = int(1/precision)
    return x_int / factor

def cantor_pairing(x_int, y_int):
    """
    Cantor 配对函数，将两个非负整数编码为一个唯一整数 z。
    """
    return ((x_int + y_int) * (x_int + y_int + 1)) // 2 + y_int

def inverse_cantor(z):
    """
    Cantor 配对函数的逆函数，从 z 解码出 x_int 和 y_int。
    """
    w = int((math.sqrt(8*z + 1) - 1) // 2)
    t = (w * (w + 1)) // 2
    y_int = z - t
    x_int = w - y_int
    return x_int, y_int

# 示例：假设原始坐标为 (x, y) = (3.141, 2.718)
x, y = 1.1, 2.718

# 1. 量化，将浮点数转为整数
x_int = quantize(x)  # 例如 3141
y_int = quantize(y)  # 例如 2718

# 2. 使用 Cantor 配对函数编码
z = cantor_pairing(x_int, y_int)
print("编码后的 z:", z)

# 3. 从 z 解码出整数坐标
x_int_decoded, y_int_decoded = inverse_cantor(z)

# 4. 反量化，恢复为浮点数
x_decoded = dequantize(x_int_decoded)
y_decoded = dequantize(y_int_decoded)
print("解码恢复的 (x, y):", (x_decoded, y_decoded))
