import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler

def crack_growth_with_factors(t, a, C0=1e-10, m=3.0, sigma=200, T=298, H2S=0, C_threshold=1e-4, alpha=0.5, beta=2, mu=0.5, pi=np.pi):
    if isinstance(a, np.ndarray):
        a = a[0]
    C_T = C0 * np.exp(-mu / (8.314 * T))
    SCC_factor = alpha * (H2S / C_threshold) ** beta if H2S > 0 else 0
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

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2_mean = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)
    
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2_mean(h), self.fc2_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

latent_dim = 5
vae = VAE(input_dim=1, latent_dim=latent_dim).cuda()
optimizer = optim.Adam(vae.parameters(), lr=0.001)
criterion = nn.MSELoss()
scaler = StandardScaler()

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 500)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

G = Generator(input_dim=10).cuda()
D = Discriminator(input_dim=500).cuda()
optim_G = optim.Adam(G.parameters(), lr=0.0002)
optim_D = optim.Adam(D.parameters(), lr=0.0002)

time_values, crack_sizes = simulate_crack_growth()

def train_vae(vae, optimizer, time_values, crack_sizes, epochs=50):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(crack_sizes)):
            optimizer.zero_grad()
            
            crack_size = torch.tensor([[crack_sizes[i]]], dtype=torch.float32).cuda()
            
            recon_batch, mu, logvar = vae(crack_size)
            
            recon_loss = criterion(recon_batch, crack_size)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(crack_sizes)}")

def train_gan(G, D, optim_G, optim_D, epochs=50):
    for epoch in range(epochs):
        optim_D.zero_grad()
        
        real_data = torch.tensor(np.random.randn(1, 500), dtype=torch.float32).cuda()
        real_labels = torch.ones(1, 1).cuda()
        outputs = D(real_data)
        d_loss_real = criterion(outputs, real_labels)
        
        z = torch.randn(1, 10).cuda()
        fake_data = G(z).detach()
        fake_labels = torch.zeros(1, 1).cuda()
        outputs = D(fake_data)
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optim_D.step()
        
        optim_G.zero_grad()
        z = torch.randn(1, 10).cuda()
        fake_data = G(z)
        outputs = D(fake_data)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optim_G.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D loss: {d_loss.item()}, G loss: {g_loss.item()}")

train_vae(vae, optimizer, time_values, crack_sizes)
train_gan(G, D, optim_G, optim_D)

def generate_synthetic_data(N=500):
    data = []
    scaler.fit(np.array(crack_sizes).reshape(-1, 1))
    
    for i in range(N):
        a = crack_sizes[i % len(crack_sizes)]
        crack_size_scaled = scaler.transform([[a]])
        crack_size_tensor = torch.tensor(crack_size_scaled, dtype=torch.float32).cuda()
        
        with torch.no_grad():
            fem_approx = vae.decode(vae.reparameterize(*vae.encode(crack_size_tensor))).cpu().numpy()[0][0]
        
        burst_noise = generate_burst_noise(500)
        z = torch.randn((1, 10), device="cuda")
        with torch.no_grad():
            wave_signal = G(z).cpu().numpy().flatten()
            wave_signal += burst_noise
            
        amplitude = np.max(np.abs(wave_signal)) * 100
        data.append([time_values[i % len(time_values)], a, fem_approx, amplitude])
    
    return np.array(data)

data = generate_synthetic_data()
dataset = pd.DataFrame(data, columns=["Time (hours)", "CrackSize", "FEM_Approx", "Amplitude (dB)"])
dataset.to_csv("hybrid_synthetic_ultrasonic_data.csv", index=False)

print("Hybrid Synthetic Data Generation Complete with Time-Dependent Crack Growth!")
