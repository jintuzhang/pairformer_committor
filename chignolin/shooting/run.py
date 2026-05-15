import os
import sys
import numpy as np

import openmm
import openmmplumed
from openmm import app
from openmm import unit
from openmmtools import integrators


def run(gro_file: str) -> None:

    top_file = '../../../../data/gmx/topol.top'

    gro = app.GromacsGroFile(gro_file)
    top = app.GromacsTopFile(
            top_file,
            periodicBoxVectors=gro.getPeriodicBoxVectors(),
            includeDir='../../../../data/gmx'
    )
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.95 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )
    with open('../../../plumed.inp', 'r') as fp:
        plumed = openmmplumed.PlumedForce(fp.read())
    system.addForce(plumed)

    integrator = integrators.GeodesicBAOABIntegrator(
        temperature=340 * unit.kelvin,
        collision_rate=10 / unit.picosecond,
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
    dcd_reporter = app.DCDReporter(
        'traj.dcd',
        reportInterval=500,
        atomSubset=list(range(166)),
    )
    screen_reporter = app.StateDataReporter(
        sys.stdout, 500, step=True, potentialEnergy=True, temperature=True
    )
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(screen_reporter)

    for _ in range(int(1.5E7) // 1000):
        simulation.step(1000)
        if os.path.exists('STOPCAR') and os.stat('STOPCAR').st_size != 0:
            del simulation
            return


if __name__ == '__main__':
    for i in np.arange(18):
        for j in np.arange(20):
            os.system(f'rm -rf ./results/{i}/{j}')
            os.mkdir(f'./results/{i}/{j}')
            os.chdir(f'./results/{i}/{j}')
            run(f'../../data/{i}.gro')
            os.chdir('../../../')
