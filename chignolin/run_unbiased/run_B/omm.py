import sys

import openmm
import openmmplumed
from openmm import app
from openmm import unit
from openmmtools import integrators


gro_file = '../../data/gmx/unfolded.gro'
top_file = '../../data/gmx/topol.top'

gro = app.GromacsGroFile(gro_file)
top = app.GromacsTopFile(
        top_file,
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir='../../data/gmx'
)
system = top.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.95 * unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
)
with open('plumed.inp', 'r') as fp:
    plumed = openmmplumed.PlumedForce(fp.read())
system.addForce(plumed)

integrator = integrators.GeodesicBAOABIntegrator(
    temperature=340 * unit.kelvin,
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
simulation.context.setVelocitiesToTemperature(340)
dcd_reporter = app.DCDReporter('traj.dcd', reportInterval=500)
screen_reporter = app.StateDataReporter(
    sys.stdout, 500, step=True, potentialEnergy=True, temperature=True
)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(screen_reporter)

simulation.step(int(5E6))
