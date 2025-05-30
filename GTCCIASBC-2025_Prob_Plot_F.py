# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZP1vPm4O8YoyeUL9ULWEaqia0joKGovn
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Physical constants
hbar = 1.0545718e-34  # J·s
m_e = 9.1093837e-31   # kg
eV = 1.60217662e-19   # J/eV
kcal_per_mol = 23.06  # eV to kcal/mol

# PIB Model Functions
def pib_wavefunction(x, n, L):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def pib_energy(n, L):
    return (hbar**2 * np.pi**2 * n**2) / (2 * m_e * L**2)

def plot_pib(L, n_max=3, pib_num_particles=1):
    print('Generating PIB plot...')
    x = np.linspace(0, L, 1000)
    fig = plt.figure(figsize=(15, 5))

    # Wavefunction plot
    ax1 = fig.add_subplot(131)
    max_energy = pib_energy(n_max, L)
    for n in range(1, n_max + 1):
        psi = pib_wavefunction(x, n, L)
        ax1.plot(x * 1e9, psi, label=f'n={n}')
    ax1.set_title('PIB Wavefunctions')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('ψ(x)')
    ax1.legend()
    ax1.grid(True)

    # Probability density plot
    ax2 = fig.add_subplot(132)
    for n in range(1, n_max + 1):
        psi = pib_wavefunction(x, n, L)
        prob_density = psi**2
        ax2.plot(x * 1e9, prob_density, label=f'n={n}')
    ax2.set_title('PIB Probability Density')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('|ψ(x)|²')
    ax2.legend()
    ax2.grid(True)

    # Energy levels plot
    ax3 = fig.add_subplot(133)
    energies = []
    particles_filled = 0
    for n in range(1, n_max + 1):
        energy = pib_energy(n, L)
        energies.append(energy)
        ax3.axhline(energy, linestyle='-', label=f'n={n}')
        if particles_filled < pib_num_particles:
            remaining = min(2, pib_num_particles - particles_filled)
            ax3.text(0.5, energy, f'{remaining} particle(s)', verticalalignment='bottom')
            particles_filled += remaining
    ax3.set_title('PIB Energy Levels')
    ax3.set_ylabel('Energy (J)')
    ax3.legend()
    if energies:
        ax3.set_ylim(0, max(energies) * 1.1)
    else:
        ax3.set_ylim(0, 1e-18)

    plt.tight_layout()
    plt.savefig('pib_visualization.png')
    plt.show()
    print('PIB plot saved as pib_visualization.png')
    return fig

# POR Model Functions
def por_wavefunction(phi, ml):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(1j * ml * phi)

def por_energy(ml, R):
    return (ml**2 * hbar**2) / (2 * m_e * R**2)

def plot_por(R, n_max=3, por_num_particles=1):
    print('Generating POR plot...')
    phi = np.linspace(0, 2 * np.pi, 1000)
    norm = 1 / np.sqrt(2 * np.pi)  # Normalization constant
    ml = n_max  # Plot only the wave function for ml = n_max
    fig = plt.figure(figsize=(15, 5))

    # Real part plot
    ax1 = fig.add_subplot(131, projection='polar')
    psi_real = np.real(por_wavefunction(phi, ml))
    ax1.plot(phi, psi_real, label='Real Part', color='blue')
    ax1.set_title(f'Re[ψ(φ)] (ml={ml})')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(-norm*1.2, norm*1.2)

    # Imaginary part plot
    ax2 = fig.add_subplot(132, projection='polar')
    psi_imag = np.imag(por_wavefunction(phi, ml))
    ax2.plot(phi, psi_imag, label='Imaginary Part', color='red')
    ax2.set_title(f'Im[ψ(φ)] (ml={ml})')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(-norm*1.2, norm*1.2)

    # Probability density plot
    ax3 = fig.add_subplot(133, projection='polar')
    probability_density = np.full_like(phi, norm**2)
    ax3.plot(phi, probability_density, label='|ψ(φ)|²', color='green')
    ax3.set_title(f'Probability Density (ml={ml})')
    ax3.grid(True)
    ax3.legend()
    ax3.set_ylim(0, norm**2 * 1.5)

    # Print energy levels for context
    print('POR Energy Levels (Wavefunction Plot):')
    energies = []
    for ml_val in range(0, n_max + 1):
        energy = por_energy(ml_val, R)
        energies.append(energy)
        print(f'  ml=±{ml_val}' if ml_val > 0 else f'  ml={ml_val}: {energy:.2e} J')

    plt.tight_layout()
    plt.savefig('por_visualization.png')
    plt.show()
    print('POR plot saved as por_visualization.png')
    return fig

# Energy Calculation
def calculate_energies(L, R, pib_num_particles=6, por_num_particles=6):
    pib_total_energy = 0
    particles_filled = 0
    n = 1
    print('PIB Energy Levels:')
    while particles_filled < pib_num_particles:
        energy = pib_energy(n, L)
        particles_to_add = min(2, pib_num_particles - particles_filled)
        pib_total_energy += particles_to_add * energy
        print(f'  n={n}: {energy:.2e} J, {particles_to_add} particle(s)')
        particles_filled += particles_to_add
        n += 1
    por_total_energy = 0
    particles_filled = 0
    ml = 0
    print('POR Energy Levels:')
    while particles_filled < por_num_particles:
        energy = por_energy(ml, R)
        particles_to_add = min(2 if ml == 0 else 4, por_num_particles - particles_filled)
        por_total_energy += particles_to_add * energy
        print(f'  ml=±{ml}' if ml > 0 else f'  ml={ml}: {energy:.2e} J, {particles_to_add} particle(s)')
        particles_filled += particles_to_add
        ml += 1
    ase = pib_total_energy - por_total_energy
    return pib_total_energy, por_total_energy, ase

# Create widgets
L_input = widgets.FloatText(value=1.03, description='L (nm):')
R_input = widgets.FloatText(value=0.139, description='R (nm):')
n_max_input = widgets.IntText(value=3, description='n_max:')
pib_particles_input = widgets.IntText(value=6, description='PIB Particles:')
por_particles_input = widgets.IntText(value=6, description='POR Particles:')
plot_button = widgets.Button(description='Calculate and Plot')

# Create output widget
output = widgets.Output()

# Display widgets and output
display(L_input, R_input, n_max_input, pib_particles_input, por_particles_input, plot_button, output)

def on_plot_button_clicked(b):
    with output:
        clear_output(wait=True)  # Clear previous plots and text
        L = L_input.value * 1e-9
        R = R_input.value * 1e-9
        n_max = n_max_input.value
        pib_num_particles = pib_particles_input.value
        por_num_particles = por_particles_input.value
        if L <= 0:
            print('Error: PIB length (L) must be positive')
            return
        if R <= 0:
            print('Error: POR radius (R) must be positive')
            return
        if n_max < 0:
            print('Error: n_max must be non-negative')
            return
        if pib_num_particles < 1 or por_num_particles < 1:
            print('Error: Number of particles must be at least 1 for both PIB and POR')
            return
        plot_pib(L, n_max, pib_num_particles)
        plot_por(R, n_max, por_num_particles)
        pib_energy, por_energy, ase = calculate_energies(L, R, pib_num_particles, por_num_particles)
        print(f'\nTotal PIB Energy ({pib_num_particles} particles): {pib_energy:.2e} J ({pib_energy/eV:.2f} eV, {pib_energy/eV*kcal_per_mol:.2f} kcal/mol)')
        print(f'Total POR Energy ({por_num_particles} particles): {por_energy:.2e} J ({por_energy/eV:.2f} eV, {por_energy/eV*kcal_per_mol:.2f} kcal/mol)')
        print(f'Aromatic Stabilization Energy: {ase:.2e} J ({ase/eV:.2f} eV, {ase/eV*kcal_per_mol:.2f} kcal/mol)')

plot_button.on_click(on_plot_button_clicked)