import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import kivy
kivy.require('1.11.1')
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Ellipse, Line, Rotate, PushMatrix, PopMatrix, Rectangle
from kivy.properties import NumericProperty
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.window import Window

# Physical constants
hbar = 1.0545718e-34
m_e = 9.1093837e-31

# PIB Model Functions
def pib_wavefunction(x, n, L):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def pib_energy(n, L):
    return (hbar**2 * np.pi**2 * n**2) / (2 * m_e * L**2)

def plot_pib(L, n_max=3, num_particles=1):
    x = np.linspace(0, L, 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    max_energy = pib_energy(n_max, L)
    for n in range(1, n_max + 1):
        psi = pib_wavefunction(x, n, L)
        energy = pib_energy(n, L)
        ax1.plot(x * 1e9, psi + energy / max_energy, label=f"n={n}")
    ax1.set_title("PIB Wavefunctions", color='white')
    ax1.set_xlabel("x (nm)", color='white')
    ax1.set_ylabel("ψ(x) + E (scaled)", color='white')
    ax1.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
    ax1.set_facecolor('#333333')
    ax1.tick_params(colors='white')
    
    energies = []
    particles_filled = 0
    for n in range(1, n_max + 1):
        energy = pib_energy(n, L)
        energies.append(energy)
        ax2.axhline(energy, color='white', linestyle='-', label=f"n={n}")
        if particles_filled < num_particles:
            remaining = min(2, num_particles - particles_filled)
            ax2.text(0.5, energy, f"{remaining} particle(s)", verticalalignment='bottom', color='white')
            particles_filled += remaining
    ax2.set_title("PIB Energy Levels", color='white')
    ax2.set_ylabel("Energy (J)", color='white')
    ax2.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
    ax2.set_facecolor('#333333')
    ax2.tick_params(colors='white')
    if energies:
        ax2.set_ylim(0, max(energies) * 1.1)
    else:
        ax2.set_ylim(0, 1e-18)
    
    fig.patch.set_facecolor('#1E1E1E')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='#1E1E1E')
    plt.close(fig)
    buf.seek(0)
    return buf

# POR Model Functions
def por_wavefunction(phi, ml):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(1j * ml * phi)

def por_energy(ml, R):
    return (ml**2 * hbar**2) / (2 * m_e * R**2)

def plot_por(R, n_max=3, num_particles=1):
    phi = np.linspace(0, 2 * np.pi, 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    max_energy = por_energy(n_max, R)
    for ml in range(0, n_max + 1):
        psi_real = np.real(por_wavefunction(phi, ml))
        energy = por_energy(ml, R)
        ax1.plot(phi, psi_real + energy / max_energy, label=f"ml={ml}")
    ax1.set_title("POR Wavefunctions (Real Part)", color='white')
    ax1.set_xlabel("φ (rad)", color='white')
    ax1.set_ylabel("Re[ψ(φ)] + E (scaled)", color='white')
    ax1.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
    ax1.set_facecolor('#333333')
    ax1.tick_params(colors='white')
    
    energies = []
    particles_filled = 0
    for ml in range(0, n_max + 1):
        energy = por_energy(ml, R)
        energies.append(energy)
        ax2.axhline(energy, color='white', linestyle='-', label=f"ml=±{ml}" if ml > 0 else "ml=0")
        if particles_filled < num_particles:
            remaining = min(2 if ml == 0 else 4, num_particles - particles_filled)
            ax2.text(0.5, energy, f"{remaining} particle(s)", verticalalignment='bottom', color='white')
            particles_filled += remaining
    ax2.set_title("POR Energy Levels", color='white')
    ax2.set_ylabel("Energy (J)", color='white')
    ax2.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
    ax2.set_facecolor('#333333')
    ax2.tick_params(colors='white')
    if energies:
        ax2.set_ylim(-max(energies) * 0.1, max(energies) * 1.1)
    else:
        ax2.set_ylim(-1e-20, 1e-20)
    
    fig.patch.set_facecolor('#1E1E1E')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='#1E1E1E')
    plt.close(fig)
    buf.seek(0)
    return buf

# Kivy Screens
class IntroScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with self.canvas.before:
            Color(0.12, 0.12, 0.12, 1)  # Dark background #1E1E1E
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        # Main label with title and subtitle
        main_label = Label(
            text="[size=24sp]Ring Quest[/size]\n[size=18sp]Explore Aromatic Molecules[/size]",
            font_name='Roboto',
            bold=True,
            color=(1, 0.44, 0.38, 1),  # #FF6F61
            size_hint=(1, 0.9),
            halign='center',
            valign='middle',
            text_size=(Window.width - 40, None),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            markup=True
        )
        layout.add_widget(main_label)
        
        self.add_widget(layout)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class PIBScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with self.canvas.before:
            Color(0.12, 0.12, 0.12, 1)  # Dark background #1E1E1E
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        self.layout.add_widget(Label(text="Box Length L (nm):", color=(1, 1, 1, 1), size_hint=(1, 0.05)))
        self.L_input = TextInput(text="1.0", multiline=False, background_color=(0.42, 0.45, 0.56, 1), size_hint=(1, 0.05))
        self.layout.add_widget(self.L_input)
        
        self.layout.add_widget(Label(text="Max Quantum Number n:", color=(1, 1, 1, 1), size_hint=(1, 0.05)))
        self.n_input = TextInput(text="3", multiline=False, background_color=(0.42, 0.45, 0.56, 1), size_hint=(1, 0.05))
        self.layout.add_widget(self.n_input)
        
        self.layout.add_widget(Label(text="Number of Particles:", color=(1, 1, 1, 1), size_hint=(1, 0.05)))
        self.particles_input = TextInput(text="1", multiline=False, background_color=(0.42, 0.45, 0.56, 1), size_hint=(1, 0.05))
        self.layout.add_widget(self.particles_input)
        
        self.plot_button = Button(text="Plot PIB", background_color=(1, 0.44, 0.38, 1), size_hint=(1, 0.05))  # #FF6F61
        self.plot_button.bind(on_press=self.plot)
        self.layout.add_widget(self.plot_button)
        
        self.image = KivyImage(size_hint=(1, 0.6))
        self.layout.add_widget(self.image)
        
        self.add_widget(self.layout)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def plot(self, instance):
        try:
            L = float(self.L_input.text) * 1e-9
            n_max = int(self.n_input.text)
            num_particles = int(self.particles_input.text)
            if L <= 0 or n_max < 1 or num_particles < 1:
                print("Error: L > 0, n ≥ 1, particles ≥ 1")
                return
            buf = plot_pib(L, n_max, num_particles)
            self.image.texture = CoreImage(buf, ext='png').texture
        except ValueError:
            print("Error: Invalid input")

class PORScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with self.canvas.before:
            Color(0.12, 0.12, 0.12, 1)  # Dark background #1E1E1E
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        self.layout.add_widget(Label(text="Ring Radius R (nm):", color=(1, 1, 1, 1), size_hint=(1, 0.05)))
        self.R_input = TextInput(text="0.139", multiline=False, background_color=(0.42, 0.45, 0.56, 1), size_hint=(1, 0.05))
        self.layout.add_widget(self.R_input)
        
        self.layout.add_widget(Label(text="Max Quantum Number ml:", color=(1, 1, 1, 1), size_hint=(1, 0.05)))
        self.n_input = TextInput(text="3", multiline=False, background_color=(0.42, 0.45, 0.56, 1), size_hint=(1, 0.05))
        self.layout.add_widget(self.n_input)
        
        self.layout.add_widget(Label(text="Number of Particles:", color=(1, 1, 1, 1), size_hint=(1, 0.05)))
        self.particles_input = TextInput(text="1", multiline=False, background_color=(0.42, 0.45, 0.56, 1), size_hint=(1, 0.05))
        self.layout.add_widget(self.particles_input)
        
        self.plot_button = Button(text="Plot POR", background_color=(1, 0.44, 0.38, 1), size_hint=(1, 0.05))
        self.plot_button.bind(on_press=self.plot)
        self.layout.add_widget(self.plot_button)
        
        self.image = KivyImage(size_hint=(1, 0.6))
        self.layout.add_widget(self.image)
        
        self.add_widget(self.layout)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def plot(self, instance):
        try:
            R = float(self.R_input.text) * 1e-9
            n_max = int(self.n_input.text)
            num_particles = int(self.particles_input.text)
            if R <= 0 or n_max < 0 or num_particles < 1:
                print("Error: R > 0, ml ≥ 0, particles ≥ 1")
                return
            buf = plot_por(R, n_max, num_particles)
            self.image.texture = CoreImage(buf, ext='png').texture
        except ValueError:
            print("Error: Invalid input")

class BenzeneScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with self.canvas.before:
            Color(0.12, 0.12, 0.12, 1)  # Dark background #1E1E1E
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        self.canvas_widget = BoxLayout()
        self.layout.add_widget(self.canvas_widget)
        
        self.angle = 0
        self.draw_benzene()
        
        self.add_widget(self.layout)
        
        Clock.schedule_interval(self.update, 1/60)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def draw_benzene(self):
        self.canvas_widget.canvas.clear()
        with self.canvas_widget.canvas:
            center_x, center_y = self.canvas_widget.center
            radius = min(self.canvas_widget.width, self.canvas_widget.height) * 0.3
            
            # Draw hexagon (benzene ring)
            Color(1, 1, 1, 1)  # White
            points = []
            for i in range(6):
                angle = np.pi/3 * i
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                points.extend([x, y])
            Line(points=points, close=True, width=2)
            
            # Draw alternating double bonds
            for i in range(0, 6, 2):
                angle1 = np.pi/3 * i
                angle2 = np.pi/3 * (i + 1)
                x1 = center_x + radius * 0.85 * np.cos(angle1)
                y1 = center_y + radius * 0.85 * np.sin(angle1)
                x2 = center_x + radius * 0.85 * np.cos(angle2)
                y2 = center_y + radius * 0.85 * np.sin(angle2)
                Line(points=[x1, y1, x2, y2], width=2)
            
            # Draw rotating arc
            PushMatrix()
            self.rotation = Rotate()
            self.rotation.origin = (center_x, center_y)
            self.rotation.angle = self.angle
            Color(0, 1, 0, 0.7)  # Green, semi-transparent
            arc_points = []
            num_points = 30
            arc_radius = radius * 0.5
            for i in range(num_points + 1):
                theta = i * (np.pi / 2) / num_points  # 90-degree arc
                x = center_x + arc_radius * np.cos(theta)
                y = center_y + arc_radius * np.sin(theta)
                arc_points.extend([x, y])
            Line(points=arc_points, width=5, cap='round', joint='round')
            PopMatrix()

    def update(self, dt):
        self.angle += 2
        self.draw_benzene()

    def on_size(self, *args):
        self.draw_benzene()

class EnergyScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with self.canvas.before:
            Color(0.12, 0.12, 0.12, 1)  # Dark background #1E1E1E
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        self.layout.add_widget(Label(text="Calculate Energies for 6 Particles", color=(1, 1, 1, 1), size_hint=(1, 0.05)))
        
        self.calc_button = Button(text="Calculate Energies", background_color=(1, 0.44, 0.38, 1), size_hint=(1, 0.05))
        self.calc_button.bind(on_press=self.calculate)
        self.layout.add_widget(self.calc_button)
        
        self.compare_button = Button(text="Compare Length and Circumference", background_color=(0.42, 0.45, 0.56, 1), size_hint=(1, 0.05))
        self.compare_button.bind(on_press=self.compare)
        self.layout.add_widget(self.compare_button)
        
        self.compare_label = Label(text="", color=(1, 1, 1, 1), size_hint=(1, 0.05))
        self.layout.add_widget(self.compare_label)
        
        self.result_label = Label(text="", color=(1, 1, 1, 1), size_hint=(1, 0.75))
        self.layout.add_widget(self.result_label)
        
        self.add_widget(self.layout)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def calculate(self, instance):
        try:
            pib_screen = self.manager.get_screen('pib')
            por_screen = self.manager.get_screen('por')
            L = float(pib_screen.L_input.text) * 1e-9
            R = float(por_screen.R_input.text) * 1e-9
            
            pib_total_energy = 0
            particles_filled = 0
            n = 1
            while particles_filled < 6:
                energy = pib_energy(n, L)
                particles_to_add = min(2, 6 - particles_filled)
                pib_total_energy += particles_to_add * energy
                particles_filled += particles_to_add
                n += 1
            
            por_total_energy = 0
            particles_filled = 0
            ml = 0
            while particles_filled < 6:
                energy = por_energy(ml, R)
                particles_to_add = min(2 if ml == 0 else 4, 6 - particles_filled)
                por_total_energy += particles_to_add * energy
                particles_filled += particles_to_add
                ml += 1
            
            ase = pib_total_energy - por_total_energy
            
            self.result_label.text = (
                f"PIB Energy (6 particles): {pib_total_energy:.2e} J\n"
                f"POR Energy (6 particles): {por_total_energy:.2e} J\n"
                f"Aromatic Stabilization Energy: {ase:.2e} J"
            )
        except ValueError:
            self.result_label.text = "Error: Invalid input"

    def compare(self, instance):
        try:
            por_screen = self.manager.get_screen('por')
            pib_screen = self.manager.get_screen('pib')
            R = float(por_screen.R_input.text) * 1e-9
            L = float(pib_screen.L_input.text) * 1e-9
            circumference = 2 * np.pi * R
            self.compare_label.text = f"Circumference: {circumference*1e9:.2f} nm\nPIB Length: {L*1e9:.2f} nm\nDifference: {(circumference-L)*1e9:.2f} nm"
        except ValueError:
            self.compare_label.text = "Error: Invalid input"

class AromaticityApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(IntroScreen(name='intro'))
        sm.add_widget(PIBScreen(name='pib'))
        sm.add_widget(PORScreen(name='por'))
        sm.add_widget(BenzeneScreen(name='benzene'))
        sm.add_widget(EnergyScreen(name='energy'))
        
        for screen in sm.screens:
            layout = screen.children[0]
            nav_layout = BoxLayout(size_hint=(1, 0.1))
            for other_screen in sm.screens:
                btn = Button(text=other_screen.name.capitalize(), background_color=(0.42, 0.45, 0.56, 1))
                btn.bind(on_press=lambda instance, s=other_screen.name: setattr(sm, 'current', s))
                nav_layout.add_widget(btn)
            layout.add_widget(nav_layout)
        
        return sm

if __name__ == '__main__':
    AromaticityApp().run()