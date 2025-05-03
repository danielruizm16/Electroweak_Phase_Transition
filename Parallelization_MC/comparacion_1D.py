import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from tqdm import tqdm

# =============================================================================
# Configuración común (parámetros físicos y funciones)
# =============================================================================
mh = 125
v0 = 246.22
lam_phi = mh**2/(2*v0**2)
lam_phi_chi = 0.5
lam_chi = 0.1
g = 0.652
gprime = 0.357
y_t = 0.99
m_DM = 80
mu_chi = m_DM**2 - lam_phi_chi*v0**2
a = (9*g**2 + 3*gprime**2 + 12*y_t**2 + 12*lam_phi + 4*lam_phi_chi)/48
b = (lam_phi_chi + 3*lam_chi)/12

N = 100
N_steps = 10000
delta = 0.5

def potential(phi, chi, T):
    return (lam_phi * (phi**2 - v0**2)**2 + 
            0.5 * mu_chi * chi**2 + 
            lam_chi * chi**4 + 
            lam_phi_chi * phi**2 * chi**2 + 
            (a*T*phi)**2 + (b*T*chi)**2)

# =============================================================================
# Versión Serial
# =============================================================================
def simulate_temperature(T, seed):
    np.random.seed(seed)
    phi = np.random.randn(N)
    chi = np.random.randn(N)
    
    for _ in range(N_steps):
        for i in range(N):
            # Actualización de φ
            phi_old = phi[i]
            phi_new = phi_old + np.random.uniform(-delta, delta)
            delta_S_phi = potential(phi_new, chi[i], T) - potential(phi_old, chi[i], T)
            if delta_S_phi < 0 or np.random.uniform() < np.exp(-delta_S_phi):
                phi[i] = phi_new
            
            # Actualización de χ
            chi_old = chi[i]
            chi_new = chi_old + np.random.uniform(-delta, delta)
            delta_S_chi = potential(phi[i], chi_new, T) - potential(phi[i], chi_old, T)
            if delta_S_chi < 0 or np.random.uniform() < np.exp(-delta_S_chi):
                chi[i] = chi_new
    
    return T, np.mean(phi), np.mean(chi)

def run_serial(temperatures, seeds):
    results = []
    for T, seed in tqdm(zip(temperatures, seeds), total=len(temperatures), desc="Serial"):
        results.append(simulate_temperature(T, seed))
    results.sort()
    return zip(*results)

# =============================================================================
# Versión Paralela
# =============================================================================
def run_parallel(temperatures, seeds):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.starmap(simulate_temperature, zip(temperatures, seeds)),
            total=len(temperatures),
            desc="Paralelo"
        ))
    results.sort()
    return zip(*results)

# =============================================================================
# Ejecución y visualización comparativa
# =============================================================================
if __name__ == '__main__':
    # Configuración común
    temperatures = np.linspace(50, 300, 20)
    seeds = np.random.randint(0, 10000, size=len(temperatures))
    
    # Ejecutar serial
    start_serial = time.time()
    temps_serial, phi_serial, chi_serial = run_serial(temperatures, seeds)
    serial_time = time.time() - start_serial
    
    # Ejecutar paralelo
    start_parallel = time.time()
    temps_parallel, phi_parallel, chi_parallel = run_parallel(temperatures, seeds)
    parallel_time = time.time() - start_parallel
    
    # Resultados de tiempos
    print(f"\nTiempo SERIAL: {serial_time:.2f} segundos")
    print(f"Tiempo PARALELO: {parallel_time:.2f} segundos")
    print(f"Speedup: {serial_time/parallel_time:.2f}x")
    
    # Visualización comparativa
    plt.figure(figsize=(14, 6))
    
    # Gráfica para φ
    plt.subplot(1, 2, 1)
    plt.plot(temps_serial, phi_serial, 'o-', color='blue', label='Serial', linewidth=2)
    plt.plot(temps_parallel, phi_parallel, 'x--', color='red', label='Paralelo', linewidth=1.5)
    plt.xlabel('Temperature (GeV)')
    plt.ylabel('$\\langle \\phi \\rangle$ (GeV)')
    plt.title('Comparación: Campo $\\phi$')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Gráfica para χ
    plt.subplot(1, 2, 2)
    plt.plot(temps_serial, chi_serial, 's-', color='blue', label='Serial', linewidth=2)
    plt.plot(temps_parallel, chi_parallel, 'd--', color='red', label='Paralelo', linewidth=1.5)
    plt.xlabel('Temperature (GeV)')
    plt.ylabel('$\\langle \\chi \\rangle$ (GeV)')
    plt.title('Comparación: Campo $\\chi$')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.suptitle('Transición de Fase Electroweak: Serial vs. Paralelo')
    plt.tight_layout()
    plt.show()