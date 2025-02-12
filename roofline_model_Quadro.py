
import matplotlib.pyplot as plt
import numpy as np

# Paramètres du modèle Roofline
bandwidth = 6.72e11  # en bytes par seconde
peak_performance = 16.31e12  # en FLOPS

# Données pour le tracé
operational_intensity = np.linspace(0.001, 10**5, 400)
bandwidth_limited_performance = bandwidth * operational_intensity

# Données des opérations
operations = [
    {"name": "Matrix multiplication", "flops": 16382, "mem": 513, "oi": 32, "flop_s": 3.447e10},
    {"name": "Add bias", "flops": 1, "mem": 3, "oi": 0.33333, "flop_s": 1.984126e6},
    {"name": "Sigmoid", "flops": 2, "mem": 2, "oi": 1, "flop_s": 1.06473594e8},
    {"name": "Softmax", "flops": 514, "mem": 258, "oi": 2, "flop_s": 1.324196208e10},
    {"name": "Compute delta1", "flops": 5, "mem": 3, "oi": 1.6667, "flop_s": 3.09405940e8},
    {"name": "Compute delta2", "flops": 1, "mem": 513, "oi": 1.949e-3, "flop_s": 2.0305393e7},
    {"name": "Compute derivative W", "flops": 16384, "mem": 514, "oi": 31.8754, "flop_s": 3.6209335e11},
    {"name": "Compute derivative b", "flops": 258, "mem": 259, "oi": 1, "flop_s": 7.390009166e9}
]
colors = plt.cm.viridis(np.linspace(0, 1, len(operations)))
# Tracé du modèle Roofline
plt.figure(figsize=(20, 12))
plt.plot(operational_intensity, bandwidth_limited_performance, label='Bandwidth limitation', color='blue')
plt.hlines(peak_performance, min(operational_intensity), max(operational_intensity), label='Peak performance', colors='red')

# Ajouter les points des opérations et les annotations
for idx, op in enumerate(operations):
    if op["flop_s"] is not None:  # Vérifie si la valeur de FLOPS est disponible
        plt.scatter(op["oi"], op["flop_s"], color=colors[idx])
        plt.annotate(op["name"], (op["oi"], op["flop_s"]),
                     textcoords="offset points", xytext=(50, 0),
                     ha='left', color=colors[idx],
                     arrowprops=dict(arrowstyle="->", linestyle="dotted", color=colors[idx]))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Operational Intensity (FLOPs/byte)')
plt.ylabel('Performance (FLOPS)')
plt.title('Roofline model - Nvidia Quadro RTX 6000')
plt.legend(loc='upper left')

plt.grid(True)
plt.show()
