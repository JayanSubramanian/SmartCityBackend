import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Create FEM surrogate model
def initialize_fem_surrogate():
    # Generate training data
    crack_sizes = np.linspace(0.001, 0.02, 20)
    fem_wave_responses = np.sin(2 * np.pi * 50 * crack_sizes)
    
    # Scale and train model
    scaler = StandardScaler()
    crack_sizes_scaled = scaler.fit_transform(crack_sizes.reshape(-1, 1))
    gp_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500)
    gp_model.fit(crack_sizes_scaled, fem_wave_responses)
    
    return gp_model, scaler

# Initialize FEM surrogate model once
gp_model, scaler = initialize_fem_surrogate()

def generate_synthetic_data(N=500):
    data = []
    for _ in range(N):
        # Generate crack size
        a = np.random.uniform(0.001, 0.02)
        
        # Calculate crack growth with environmental factors
        C0 = 1e-10
        m = 3.0
        sigma = 200
        T = 298 + np.random.normal(0, 10)
        H2S = np.random.uniform(0, 0.001)
        C_threshold = 1e-4
        alpha = 0.5
        beta = 2
        mu = 0.5
        pi = np.pi
        
        # Integrated crack growth calculations
        C_T = C0 * np.exp(-mu / (8.314 * T))
        SCC_factor = alpha * (H2S / C_threshold) ** beta
        C_x = np.random.normal(1.0, 0.1)
        growth_rate = C_T * (sigma * np.sqrt(pi * a)) ** m * (1 + SCC_factor) * C_x
        
        # Calculate FEM approximation using the surrogate model
        crack_size_scaled = scaler.transform([[a]])
        fem_approx = gp_model.predict(crack_size_scaled)[0]
        
        # Generate signal and noise components
        burst_probability = 0.1
        burst_strength = 10
        signal_length = 500
        
        # Create wave signal with integrated noise
        base_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, signal_length))
        noise = np.random.normal(0, 1, signal_length)
        bursts = np.random.rand(signal_length) < burst_probability
        noise[bursts] += np.random.normal(0, burst_strength, np.sum(bursts))
        wave_signal = base_signal + 0.5 * noise
        
        # Calculate pressure and flow parameters
        P1 = 1e5
        P2 = 9e4
        rho = 1000
        Cd = 0.62
        A_ref = 1e-4
        A = A_ref * (a / 0.01)
        velocity = np.sqrt(2 * (P1 - P2) / rho)
        flow_rate = Cd * A * velocity
        delta_P = rho * (velocity ** 2) / 2
        
        # Calculate spectral features
        freqs, psd = welch(wave_signal, fs=500)
        psd_norm = psd / np.sum(psd)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))  # Added small value to avoid log(0)
        
        # Calculate derived acoustic emission parameters
        amplitude = np.max(np.abs(wave_signal)) * 100
        duration = max(0.5, min(5, a * 1000))
        rise_time = duration / 10
        counts = max(5, min(50, int(flow_rate * 1e6)))
        energy = 0.5 * amplitude * amplitude * duration
        peak_frequency = max(10, min(100, 50 + a * 1000))
        rms_voltage = amplitude / 10
        
        # Material and geometry factors
        material_factor = np.random.normal(1.0, 0.1)
        crack_direction_factor = np.cos(np.random.uniform(0, np.pi))
        signal_attenuation = max(0.1, min(5, a * 100 * material_factor * crack_direction_factor))
        
        # Store all parameters
        data.append([a, fem_approx, amplitude, duration, rise_time, counts, energy, 
                    peak_frequency, rms_voltage, signal_attenuation, delta_P, 
                    flow_rate, entropy, growth_rate])
    
    return np.array(data)

# Generate data
data = generate_synthetic_data()
dataset = pd.DataFrame(data, columns=["CrackSize", "FEM_Approx", "Amplitude (dB)", 
                                     "Duration (ms)", "Rise Time (ms)", "Counts", 
                                     "Energy (a.u.)", "Peak Frequency (kHz)", 
                                     "RMS Voltage (V)", "Signal Attenuation (dB/m)", 
                                     "PressureDrop", "FlowRate", "SpectralEntropy",
                                     "GrowthRate"])

# Save to CSV
dataset.to_csv("hybrid_synthetic_ultrasonic_data.csv", index=False)

# Visualize results
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(data[:, 0], data[:, 2], label='Amplitude vs Crack Size', color='r')
axs[0].set_xlabel("Crack Size (m)")
axs[0].set_ylabel("Amplitude (dB)")
axs[0].set_title("Amplitude vs Crack Size")
axs[0].legend()

axs[1].plot(data[:, 0], data[:, 10], label='Pressure Drop vs Crack Size', color='b')
axs[1].set_xlabel("Crack Size (m)")
axs[1].set_ylabel("Pressure Drop (Pa)")
axs[1].set_title("Pressure Drop vs Crack Size")
axs[1].legend()

axs[2].plot(data[:, 0], data[:, 7], label='Peak Frequency vs Crack Size', color='g')
axs[2].set_xlabel("Crack Size (m)")
axs[2].set_ylabel("Peak Frequency (kHz)")
axs[2].set_title("Peak Frequency vs Crack Size")
axs[2].legend()

plt.tight_layout()
plt.show()

print("Consolidated Hybrid Synthetic Data Generation Complete!")