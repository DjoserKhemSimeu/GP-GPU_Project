import matplotlib.pyplot as plt
import numpy as np


bandwidth = 6.72e11  # en bytes par seconde
peak_performance = 16.31e12  # en FLOPS
computational_intensity = 1143.055074


operational_intensity = np.linspace(0.01, 10**5, 400)


bandwidth_limited_performance = bandwidth * operational_intensity


plt.figure(figsize=(10, 6))
plt.plot(operational_intensity, bandwidth_limited_performance, label='Bandwidth limitation', color='blue')
plt.hlines(peak_performance, min(operational_intensity), max(operational_intensity), label='Peak performance', colors='red')


plt.xscale('log')
plt.yscale('log')
plt.xlabel('Operational Intensity (FLOPs/byte)')
plt.ylabel('Performance (FLOPS)')
plt.title('Roofline model - Nvidia Quadro RTX 6000')
plt.legend()


plt.grid(True)
plt.show()
