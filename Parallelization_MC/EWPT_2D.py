import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm

# =============================================================================
# Configuración de parámetros físicos
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

# Coeficientes para términos de temperatura
a = (9*g**2 + 3*gprime**2 + 12*y_t**2 + 12*lam_phi + 4*lam_phi_chi)/48
b = (lam_phi_chi + 3*lam_chi)/12

# Parámetros de la simulación
L = 30  # Tamaño de la red 2D (LxL)
N_steps = 20000  # Reducido para pruebas (aumentar para producción)
delta = 0.3  # Tamaño del paso para actualización de campos
dx = 1.0  # Espaciado de la red

# =============================================================================
# Funciones para el potencial y la acción
# =============================================================================
def potential(phi, chi, T):
    """Potencial termal incluyendo términos de temperatura"""
    return (lam_phi * (phi**2 - v0**2)**2 + 
            0.5 * mu_chi * chi**2 + 
            lam_chi * chi**4 + 
            lam_phi_chi * phi**2 * chi**2 + 
            (a*T*phi)**2 + (b*T*chi)**2)

def gradient_energy(phi, chi, dx):
    """Calcula la energía de gradiente para los campos en 2D"""
    grad_phi = (np.roll(phi, -1, axis=0) - 2*phi + np.roll(phi, 1, axis=0) + 
               np.roll(phi, -1, axis=1) - 2*phi + np.roll(phi, 1, axis=1))
    grad_chi = (np.roll(chi, -1, axis=0) - 2*chi + np.roll(chi, 1, axis=0) + 
               np.roll(chi, -1, axis=1) - 2*chi + np.roll(chi, 1, axis=1))
    return 0.5 * (grad_phi**2 + grad_chi**2) / dx**2

def total_action(phi, chi, T, dx):
    """Acción total incluyendo potencial y términos de gradiente"""
    return potential(phi, chi, T) + gradient_energy(phi, chi, dx)

# =============================================================================
# Simulación para una temperatura dada (versión paralela)
# =============================================================================
def simulate_temperature(args):
    """Función modificada para aceptar un solo argumento (tuple)"""
    T, seed = args
    np.random.seed(seed)
    # Inicializar campos 2D aleatorios
    phi = np.random.randn(L, L)
    chi = np.random.randn(L, L)
    
    for _ in range(N_steps):
        # Actualización de los campos en toda la red
        for i in range(L):
            for j in range(L):
                # Actualización de φ
                phi_old = phi[i,j]
                phi_new = phi_old + np.random.uniform(-delta, delta)
                
                # Calculamos la diferencia de acción solo para el punto (i,j)
                S_old = (potential(phi[i,j], chi[i,j], T) + 
                        (0.5/dx**2) * (
                            (phi[i,j] - phi[(i+1)%L,j])**2 + 
                             (phi[i,j] - phi[(i-1)%L,j])**2 +
                            (phi[i,j] - phi[i,(j+1)%L])**2 + 
                             (phi[i,j] - phi[i,(j-1)%L])**2
                        ))
                
                S_new = (potential(phi_new, chi[i,j], T) + 
                        (0.5/dx**2) * (
                            (phi_new - phi[(i+1)%L,j])**2 + 
                             (phi_new - phi[(i-1)%L,j])**2 +
                            (phi_new - phi[i,(j+1)%L])**2 + 
                             (phi_new - phi[i,(j-1)%L])**2
                        ))
                
                delta_S_phi = S_new - S_old
                
                if delta_S_phi < 0 or np.random.random() < np.exp(-delta_S_phi):
                    phi[i,j] = phi_new
                
                # Actualización de χ (misma lógica que para φ)
                chi_old = chi[i,j]
                chi_new = chi_old + np.random.uniform(-delta, delta)
                
                S_old_chi = (potential(phi[i,j], chi[i,j], T) + 
                             (0.5/dx**2) * (
                                 (chi[i,j] - chi[(i+1)%L,j])**2 + 
                                  (chi[i,j] - chi[(i-1)%L,j])**2 +
                                 (chi[i,j] - chi[i,(j+1)%L])**2 + 
                                  (chi[i,j] - chi[i,(j-1)%L])**2
                             ))
                
                S_new_chi = (potential(phi[i,j], chi_new, T) + 
                            (0.5/dx**2) * (
                                (chi_new - chi[(i+1)%L,j])**2 + 
                                 (chi_new - chi[(i-1)%L,j])**2 +
                                (chi_new - chi[i,(j+1)%L])**2 + 
                                 (chi_new - chi[i,(j-1)%L])**2
                            ))
                
                delta_S_chi = S_new_chi - S_old_chi
                
                if delta_S_chi < 0 or np.random.random() < np.exp(-delta_S_chi):
                    chi[i,j] = chi_new
    
    # Devolver los valores medios de los campos
    return T, np.mean(phi), np.mean(chi)

def run_simulation(temperatures):
    """Ejecuta la simulación en paralelo para todas las temperaturas"""
    seeds = np.random.randint(0, 10000, size=len(temperatures))
    
    # Preparamos los argumentos como una lista de tuplas
    args_list = list(zip(temperatures, seeds))
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Usamos imap con chunksize para mejor manejo de la barra de progreso
        results = list(tqdm(
            pool.imap(simulate_temperature, args_list, chunksize=1),
            total=len(temperatures),
            desc="Progreso de la simulación",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Tiempo: {elapsed}<{remaining}]'
        ))
    
    # Ordenar resultados por temperatura
    results.sort()
    return zip(*results)

# =============================================================================
# Ejecución y visualización
# =============================================================================
if __name__ == '__main__':
    print("Iniciando simulación de transición de fase electrodébil...")
    print(f"Tamaño de la red: {L}x{L}")
    print(f"Pasos de Monte Carlo: {N_steps}")
    print(f"Núcleos disponibles: {mp.cpu_count()}")
    
    # Rango de temperaturas a simular
    temperatures = np.linspace(50, 300, 10)  # Reducido para pruebas
    
    # Ejecutar simulación
    print("\nEjecutando simulaciones...")
    temps, phi_avg, chi_avg = run_simulation(temperatures)
    
    # Visualización de resultados
    print("\nMostrando resultados...")
    plt.figure(figsize=(12, 5))
    
    # Gráfica para φ
    plt.subplot(1, 2, 1)
    plt.plot(temps, phi_avg, 'o-', color='blue', linewidth=2)
    plt.xlabel('Temperature (GeV)')
    plt.ylabel('$\\langle \\phi \\rangle$ (GeV)')
    plt.title('Valor esperado del campo $\\phi$')
    plt.grid(alpha=0.3)
    
    # Gráfica para χ
    plt.subplot(1, 2, 2)
    plt.plot(temps, chi_avg, 's-', color='red', linewidth=2)
    plt.xlabel('Temperature (GeV)')
    plt.ylabel('$\\langle \\chi \\rangle$ (GeV)')
    plt.title('Valor esperado del campo $\\chi$')
    plt.grid(alpha=0.3)
    
    plt.suptitle('Transición de Fase Electroweak en 2D con Acción Completa')
    plt.tight_layout()
    plt.show()