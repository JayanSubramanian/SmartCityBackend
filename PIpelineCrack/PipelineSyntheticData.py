import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def crack_growth_with_factors(t, a, C0=1e-10, m=3.0, sigma=200, T=298, H2S=0, C_threshold=1e-4, alpha=0.5, beta=2, mu=0.5, pi=np.pi):
    C_T = C0 * np.exp(-mu / (8.314 * T))
    SCC_factor = alpha * (H2S / C_threshold) ** beta
    C_x = np.random.normal(1.0, 0.1)
    return C_T * (sigma * np.sqrt(pi * a)) ** m * (1 + SCC_factor) * C_x

def simulate_crack_growth(a0=0.005, time_span=(0, 1000), time_eval=np.linspace(0, 1000, 500)):
    sol = solve_ivp(crack_growth_with_factors, time_span, [a0], t_eval=time_eval, method='RK45')
    return sol.t, sol.y[0]

def generate_burst_noise(N, probability=0.1, strength=10):
    noise = np.random.normal(0, 1, N)
    bursts = np.random.rand(N) < probability
    noise[bursts] += np.random.normal(0, strength, np.sum(bursts))
    return noise

def generate_fem_data(samples=20):
    crack_sizes = np.linspace(0.001, 0.02, samples)
    fem_wave_responses = np.sin(2 * np.pi * 50 * crack_sizes)
    return crack_sizes.reshape(-1, 1), fem_wave_responses

fem_inputs, fem_outputs = generate_fem_data()
scaler = StandardScaler()
fem_inputs_scaled = scaler.fit_transform(fem_inputs)
gp_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500)
gp_model.fit(fem_inputs_scaled, fem_outputs)

def predict_fem(crack_size):
    crack_size_scaled = scaler.transform([[crack_size]])
    return gp_model.predict(crack_size_scaled)[0]

def leakage_pressure_drop(crack_size, P1=1e5, P2=9e4, rho=1000, Cd=0.62, A_ref=1e-4):
    A = A_ref * (crack_size / 0.01)
    velocity = np.sqrt(2 * (P1 - P2) / rho)
    flow_rate = Cd * A * velocity
    delta_P = rho * (velocity ** 2) / 2
    return delta_P, flow_rate

def compute_spectral_entropy(signal, fs=500):
    freqs, psd = welch(signal, fs=fs)
    psd_norm = psd / np.sum(psd)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    return entropy

time_values, crack_sizes = simulate_crack_growth()

def generate_synthetic_data(N=500):
    data = []
    for i in range(N):
        a = crack_sizes[i % len(crack_sizes)]
        fem_approx = predict_fem(a)
        burst_noise = generate_burst_noise(500)
        wave_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 500)) + 0.5 * burst_noise
        delta_P, flow_rate = leakage_pressure_drop(a)
        entropy = compute_spectral_entropy(wave_signal)
        amplitude = np.max(np.abs(wave_signal)) * 100
        duration = max(0.5, min(5, a * 1000))
        rise_time = duration / 10
        counts = max(5, min(50, int(flow_rate * 1e6)))
        energy = 0.5 * amplitude * amplitude * duration
        peak_frequency = max(10, min(100, 50 + a * 1000))
        rms_voltage = amplitude / 10
        material_factor = np.random.normal(1.0, 0.1)
        crack_direction_factor = np.cos(np.random.uniform(0, np.pi))
        signal_attenuation = max(0.1, min(5, a * 100 * material_factor * crack_direction_factor))
        data.append([time_values[i % len(time_values)], a, fem_approx, amplitude, duration, rise_time, counts, energy, peak_frequency, rms_voltage, signal_attenuation, delta_P, flow_rate, entropy])
    return np.array(data)

data = generate_synthetic_data()
dataset = pd.DataFrame(data, columns=["Time (hours)", "CrackSize", "FEM_Approx", "Amplitude (dB)", "Duration (ms)", "Rise Time (ms)", "Counts", "Energy (a.u.)", "Peak Frequency (kHz)", "RMS Voltage (V)", "Signal Attenuation (dB/m)", "PressureDrop", "FlowRate", "SpectralEntropy"])
dataset.to_csv("hybrid_synthetic_ultrasonic_data.csv", index=False)

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(data[:, 1], data[:, 3], label='Amplitude vs Crack Size', color='r')
axs[0].set_xlabel("Crack Size (m)")
axs[0].set_ylabel("Amplitude (dB)")
axs[0].set_title("Amplitude vs Crack Size")
axs[0].legend()

axs[1].plot(data[:, 1], data[:, 10], label='Pressure Drop vs Crack Size', color='b')
axs[1].set_xlabel("Crack Size (m)")
axs[1].set_ylabel("Pressure Drop (Pa)")
axs[1].set_title("Pressure Drop vs Crack Size")
axs[1].legend()

axs[2].plot(data[:, 1], data[:, 8], label='Peak Frequency vs Crack Size', color='g')
axs[2].set_xlabel("Crack Size (m)")
axs[2].set_ylabel("Peak Frequency (kHz)")
axs[2].set_title("Peak Frequency vs Crack Size")
axs[2].legend()

plt.tight_layout()
plt.show()

print("Hybrid Synthetic Data Generation Complete with Crack Growth Simulation!")
