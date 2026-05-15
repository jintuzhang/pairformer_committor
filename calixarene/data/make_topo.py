import mdtraj as md


for i in [2, 5, 6]:
    gro_file = f'G{i}/b.gro'
    t = md.load(gro_file, top=gro_file)
    for a in t.top.atoms:
        if a.residue.name == 'MOL':
            a.residue.name = f'G{i}'
        if a.name == 'Na+':
            a.element = md.core.element.sodium
    t.save_pdb(f'G{i}/plumed_topo.pdb')
    for a in t.top.atoms:
        a.name = a.element.symbol
    t.save_pdb(f'G{i}/plumed_topo_pairformer.pdb')
