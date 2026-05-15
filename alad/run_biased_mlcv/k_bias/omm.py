import sys
import time

import openmm
import openmmplumed
from openmm import app
from openmm import unit
from openmmtools import integrators


gro_file = '../../data/B.gro'
top_file = '../../data/topol.top'

gro = app.GromacsGroFile(gro_file)
top = app.GromacsTopFile(
        top_file,
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir='../../data'
)
system = top.createSystem(
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds
)
with open('plumed.inp', 'r') as fp:
    plumed = openmmplumed.PlumedForce(fp.read())
system.addForce(plumed)

integrator = integrators.GeodesicBAOABIntegrator(
    K_r=4,
    temperature=300 * unit.kelvin,
    collision_rate=1 / unit.picosecond,
    timestep=0.002 * unit.picoseconds
)

platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
simulation = app.Simulation(
    top.topology, system, integrator, platform, properties
)

simulation.context.setPositions(gro.positions)
simulation.currentStep = 1  # for sync. with plumed
dcd_reporter = app.DCDReporter('traj.dcd', reportInterval=500)
screen_reporter = app.StateDataReporter(
    sys.stdout, 500, step=True, potentialEnergy=True, temperature=True
)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(screen_reporter)

n_steps = int(2E6)

start = time.time()
simulation.step(n_steps)
end = time.time()

print('speed (ns/d): ', 86400 / ((end - start) / n_steps) * 2 / 1000 / 1000)
