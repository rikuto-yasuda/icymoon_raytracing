import matplotlib.pyplot as plt
import numpy as np

Freq_str = "100000"
for i in range(-50, 70, 20):
    n = str(i)
    N = i

    filename = np.genfromtxt(
        "../../result_for_yasudaetal2022/others/benchmark_1/ray-Ptest_simple-Mtest_simple-benchmark-LO-DEG"
        + n
        + "-FREQ"
        + Freq_str
    )

    filename = np.genfromtxt(
        "../../result_for_yasudaetal2022/others/testbench_1/ray-Ptest_simple-Mtest_simple-benchmark-LO-DEG"
        + n
        + "-FREQ"
        + Freq_str
    )

    x1 = filename[:, [1]]
    # 今回は地面の位置が高度12800kmで置かれており、プロットでは地面での高度0になっているため
    z1 = filename[:, [3]] - 12800
    plt.plot(x1, z1, color="red", linewidth=0.5)

plt.xlabel("x (km)")
plt.ylabel("z (km)")
plt.title("benchmark_1_Freq" + Freq_str + "Hz")
plt.xlim(0, 10000)
plt.ylim(-2000, 10000)
plt.show()
