import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# Paris' Law with Temperature, Stress Corrosion, and Material Heterogeneity
def crack_growth_with_factors(t, a, C0=1e-10, m=3.0, sigma=200, T=298, H2S=0, C_threshold=1e-4, alpha=0.5, beta=2, mu=0.5, pi=np.pi):
    # Temperature effect (Arrhenius-like equation for temperature dependence)
    C_T = C0 * np.exp(-mu / (8.314 * T))  # Adjust C based on temperature

    # Stress corrosion effect (H2S concentration)
    SCC_factor = alpha * (H2S / C_threshold) ** beta

    # Material heterogeneity effect (Random spatial variation in material properties)
    C_x = np.random.normal(1.0, 0.1)  # Simulate random variations in material properties

    # Crack growth rate considering all factors
    return C_T * (sigma * np.sqrt(pi * a)) ** m * (1 + SCC_factor) * C_x

def normal_pipe(t, a):
    return 0  # No crack growth

# Initial crack length and simulation time
a0 = 0.005  # Initial crack length in meters
time_span = (0, 1000)  # Time in hours
time_eval = np.linspace(*time_span, 500)  # Time points for evaluation

# Solve ODE for normal and cracked pipes with refined model
sol_cracked_with_factors = solve_ivp(crack_growth_with_factors, time_span, [a0], t_eval=time_eval, method='RK45')
sol_normal = solve_ivp(normal_pipe, time_span, [a0], t_eval=time_eval, method='RK45')

def generate_burst_noise(N, burst_probability=0.1, burst_strength=10):
    # Generate burst noise with given probability and strength
    burst_signal = np.random.normal(0, 1, N)
    bursts = np.random.rand(N) < burst_probability
    burst_signal[bursts] += np.random.normal(0, burst_strength, np.sum(bursts))
    return burst_signal

time_signal = np.linspace(0, 1, 500)  # 1 second of data at 500 Hz
# Generate burst noise for cracked pipe signal
N = 500  # Number of samples
burst_noise_cracked = generate_burst_noise(N, burst_probability=0.1, burst_strength=10)
# Example: Add burst noise to the acoustic signal of the cracked pipe
acoustic_signal_cracked = np.sin(2 * np.pi * 50 * time_signal) + 0.5 * burst_noise_cracked
acoustic_signal_normal = np.sin(2 * np.pi * 50 * time_signal) + 0.1 * np.random.randn(500)

def leakage_pressure_drop(time, Q, k=1e-12, A=1e-4, mu=0.001, dPdx=1e5, crack_factor=1):
    return crack_factor * (k * A / mu * dPdx)  # Without random noise for consistency

leakage_data_normal = [leakage_pressure_drop(t, Q=0.01, crack_factor=1) for t in time_eval]
leakage_data_cracked = [leakage_pressure_drop(t, Q=0.01, crack_factor=10) for t in time_eval]  # More leakage

def generate_synthetic_data(class_label, crack_growth, acoustic_signal, leakage_data):
    data = []
    for i in range(len(crack_growth)):
        amplitude = np.abs(acoustic_signal[i]) * 100  # dB scale approximation
        duration = max(0.5, min(5, crack_growth[i] * 1000))  # Scaled from crack growth rate
        rise_time = max(0.1, duration / 10)  # Proportional to duration
        counts = max(5, min(50, int(leakage_data[i] * 1e6)))  # Related to leakage rate
        energy = amplitude * duration
        peak_frequency = max(10, min(100, 50 + crack_growth[i] * 1000))  # Frequency shift with damage
        rms_voltage = amplitude / 10
        # Example: Introduce variation in attenuation based on material heterogeneity and crack geometry
        material_factor = np.random.normal(1.0, 0.1)  # Random factor to simulate material variability
        crack_direction_factor = np.cos(np.random.uniform(0, np.pi))  # Simulate directional effect on signal
        signal_attenuation = max(0.1, min(5, crack_growth[i] * 100 * material_factor * crack_direction_factor))
        data.append([amplitude, duration, rise_time, counts, energy, peak_frequency, rms_voltage, signal_attenuation, class_label])
    return data

# Create datasets for normal and cracked pipes
normal_data = generate_synthetic_data(0, sol_normal.y[0], acoustic_signal_normal, leakage_data_normal)
cracked_data = generate_synthetic_data(1, sol_cracked_with_factors.y[0], acoustic_signal_cracked, leakage_data_cracked)

# Combine and save dataset
dataset = pd.DataFrame(normal_data + cracked_data, columns=["Amplitude (dB)", "Duration (ms)", "Rise Time (ms)", "Counts", "Energy (a.u.)", "Peak Frequency (kHz)", "RMS Voltage (V)", "Signal Attenuation (dB/m)", "Class Label"])
dataset.to_csv("synthetic_pipeline_crack_data.csv", index=False)

# --- Plot Results ---
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot Crack Growth
t, a_cracked = sol_cracked_with_factors.t, sol_cracked_with_factors.y[0]
t, a_normal = sol_normal.t, sol_normal.y[0]
axs[0].plot(t, a_cracked, label='Cracked Pipe', color='r')
axs[0].plot(t, a_normal, label='Normal Pipe', linestyle='dashed', color='b')
axs[0].set_xlabel("Time (hours)")
axs[0].set_ylabel("Crack Length (m)")
axs[0].set_title("Crack Growth Over Time")
axs[0].legend()

axs[1].plot(time_signal, acoustic_signal_normal, label='Normal Pipe', color='b')
axs[1].plot(time_signal, acoustic_signal_cracked, label='Cracked Pipe', color='r', linestyle='dashed')
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Amplitude")
axs[1].set_title("Simulated Noisy Acoustic Waveform")
axs[1].legend()



plt.tight_layout()
plt.show()
