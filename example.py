# from artacs import CombKernel
# import numpy as np
# import time

# fs = 1000
# duration = 1
# t = np.linspace(0, duration, duration * fs)
# data = np.atleast_2d(np.sin(2 * 10 * np.pi * t))

# ck = CombKernel(
#     freq=10, fs=1000, width=1, left_mode="uniform", right_mode="none"
# )

# T = []
# for i in range(10_000):
#     t0 = time.time_ns()
#     out = ck.apply(data)
#     T.append((time.time_ns() - t0) / 1000)
# print(
#     f"99% CI of calculation time for 1s data is between {np.percentile(T, 0.5):3.2f} to  {np.percentile(T, 99.5):3.2f} µs"
# )
# import matplotlib.pyplot as plt

# plt.plot(data.T, label="Raw")
# plt.plot(out.T, label="Filtered")
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from artacs import CombKernel

np.random.seed(0)

# Signal parameters
signal_length = 2000
clean_signal = np.sin(2 * np.pi * 0.1 * np.arange(signal_length))

# Introduce noise to the entire signal
noise = 0.5 * np.random.randn(signal_length)
noisy_signal = clean_signal + noise

# Apply comb filter
ck = CombKernel(freq=10, fs=1000, width=1, left_mode="gauss", right_mode="exp")
filtered_signal = ck.apply(np.atleast_2d(noisy_signal))[0]

# Visualization
t = np.arange(signal_length)
plt.figure(figsize=(12, 6))
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.7)
plt.plot(t, clean_signal, label="Clean Signal")
plt.plot(t, filtered_signal, label="Filtered Signal")  # Solid line
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Comb Filter Application on Noisy Signal')
plt.grid(True)
plt.show()


print(np.sqrt(np.mean((clean_signal-filtered_signal)**2)))
