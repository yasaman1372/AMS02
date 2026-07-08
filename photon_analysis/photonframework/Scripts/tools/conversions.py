
import numpy as np

from .constants import NAF_INDEX, AGL_INDEX, NUCLEON_MASS, MC_PARTICLE_IDS, MC_PARTICLE_CHARGES, MC_PARTICLE_MASSES, MC_PARTICLE_NUCLEON_NUMBERS

def calc_mass(beta, rigidity, charge):
    return np.sqrt(charge**2 * rigidity**2 * (1 - beta**2) / (beta**2 + 1e-10))

def calc_rig(beta, mass, charge):
    return np.sqrt(mass**2 / charge**2 * beta**2 / (1 - beta**2))

def calc_beta(rigidity, mass, charge):
    rqm = (rigidity * charge / mass)**2
    return (rqm / (rqm + 1))**0.5


def calculate_mass_resolution(rigidity, charge, mass, rigidity_resolution, beta_resolution):
    sigma_R = rigidity_resolution * rigidity
    sigma_beta = beta_resolution
    return np.sqrt((rigidity**4 * charge**4 / mass**2 + 3 * rigidity**2 * charge**2 + 3 * mass**2 + mass**4 / (rigidity**2 * charge**2)) * sigma_beta**2 + mass**2 / rigidity**2 * sigma_R**2)


class Species:
    def __init__(self, charge, mass, nucleon_number):
        self.charge = charge
        self.mass = mass
        self.nucleon_number = nucleon_number

    @staticmethod
    def from_mc_particle_id(mc_particle_id):
        return Species(MC_PARTICLE_CHARGES[mc_particle_id], MC_PARTICLE_MASSES[mc_particle_id], MC_PARTICLE_NUCLEON_NUMBERS[mc_particle_id])

SPECIES = {
    mc_particle_name: Species.from_mc_particle_id(mc_particle_id)
    for mc_particle_name, mc_particle_id in MC_PARTICLE_IDS.items()
}

class KinematicVariable:
    def __init__(self, name, to_momentum, from_momentum, dx_dp, dp_dx):
        self.name = name
        self.to_momentum = to_momentum
        self.from_momentum = from_momentum
        self.dx_dp = dx_dp
        self.dp_dx = dp_dx

    def convert_to(self, other, values, species):
        return other.from_momentum(self.to_momentum(values, species), species)

    def convert_flux_to(self, other, values_x, values_y, species):
        return values_y * self.dx_dp(self.to_momentum(values_x, species), species) * other.dp_dx(self.convert_to(other, values_x, species), species)


def momentum_to_momentum(momentum, species):
    return momentum

def dp_dp(momentum, species):
    return np.ones_like(momentum)

def rigidity_to_momentum(rigidity, species):
    return rigidity * species.charge

def momentum_to_rigidity(momentum, species):
    return momentum / species.charge

def dR_dp(momentum, species):
    return 1 / species.charge * np.ones_like(momentum)

def dp_dR(rigidity, species):
    return species.charge * np.ones_like(rigidity)

def energy_to_momentum(energy, species):
    return np.sqrt(energy**2 - species.mass**2)

def momentum_to_energy(momentum, species):
    return np.sqrt(momentum**2 + species.mass**2)

def dE_dp(momentum, species):
    return 1 / np.sqrt(1 + (species.mass / momentum)**2)

def dp_dE(energy, species):
    return 1 / np.sqrt(1 - (species.mass / energy)**2)

def kinetic_energy_to_energy(kinetic_energy, species):
    return kinetic_energy + species.mass

def energy_to_kinetic_energy(energy, species):
    return energy - species.mass

def kinetic_energy_to_momentum(kinetic_energy, species):
    return energy_to_momentum(kinetic_energy_to_energy(kinetic_energy, species), species)

def momentum_to_kinetic_energy(momentum, species):
    return energy_to_kinetic_energy(momentum_to_energy(momentum, species), species)

def dEkin_dp(momentum, species):
    return 1 / np.sqrt(1 + (species.mass / momentum)**2)

def dp_dEkin(kinetic_energy, species):
    return (kinetic_energy + species.mass) / np.sqrt((kinetic_energy + species.mass)**2 - species.mass**2)

def kinetic_energy_per_nucleon_to_momentum(kinetic_energy_per_nucleon, species):
    return kinetic_energy_to_momentum(kinetic_energy_per_nucleon * species.nucleon_number, species)

def momentum_to_kinetic_energy_per_nucleon(momentum, species):
    return momentum_to_kinetic_energy(momentum, species) / species.nucleon_number

def dEkinn_dp(momentum, species):
    return dEkin_dp(momentum, species) / species.nucleon_number

def dp_dEkinn(kinetic_energy_per_nucleon, species):
    return dp_dEkin(kinetic_energy_per_nucleon * species.nucleon_number, species) * species.nucleon_number

Momentum = KinematicVariable("Momentum", momentum_to_momentum, momentum_to_momentum, dp_dp, dp_dp)
Rigidity = KinematicVariable("Rigidity", rigidity_to_momentum, momentum_to_rigidity, dR_dp, dp_dR)
Energy = KinematicVariable("Energy", energy_to_momentum, momentum_to_energy, dE_dp, dp_dE)
KineticEnergy = KinematicVariable("KineticEnergy", kinetic_energy_to_momentum, momentum_to_kinetic_energy, dEkin_dp, dp_dEkin)
KineticEnergyPerNucleon = KinematicVariable("KineticEnergyPerNucleon", kinetic_energy_per_nucleon_to_momentum, momentum_to_kinetic_energy_per_nucleon, dEkinn_dp, dp_dEkinn)


if __name__ == "__main__":
    R = np.logspace(0, 2)
    flux = R**(-2.7)
    print(R)
    print(Rigidity.convert_to(KineticEnergyPerNucleon, R, SPECIES["He4"]))
    print(flux)
    print(Rigidity.convert_flux_to(KineticEnergyPerNucleon, R, flux, SPECIES["He4"]))

