import os
import mdtraj as md
from openmm import app


for i in [2, 5, 6]:
    for j in ['b', 'u']:

        gro_file = f'G{i}/{j}.gro'
        top_file = f'G{i}/topol.top'

        gro = app.GromacsGroFile(gro_file)
        top = app.GromacsTopFile(
            top_file, periodicBoxVectors=gro.getPeriodicBoxVectors(),
        )
        app.PDBFile.writeFile(
            top.topology, gro.positions, open('tmp.pdb', 'w'), keepIds=True
        )

        t = md.load(gro_file, top='tmp.pdb')
        t = t.make_molecules_whole()
        t.save_gro('tmp.gro')

        t = md.load('tmp.gro', top=gro_file)
        t.save_gro('tmp.gro')

        os.system('mv tmp.gro ' + gro_file)
        os.system('rm tmp.pdb')
