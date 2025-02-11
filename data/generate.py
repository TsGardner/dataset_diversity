import numpy as np
from ase.build import bulk, make_supercell, surface
from ase.calculators.emt import EMT
from ase.io import write
from calorine.tools import relax_structure
from hiphive.structure_generation import generate_mc_rattled_structures
import os
import ase

prototype_structures = {}
prototype_structures['BaZrS3_Pnma']   = ase.io.read("BaZrS3_Pnma_geometry.in", format='aims')
prototype_structures['BaZrS3_Pm3m']   = ase.io.read("BaZrS3_Pm3m_geometry.in", format='aims')
prototype_structures['BaZrS3_I4_mcm']   = ase.io.read("BaZrS3_I4_mcm_geometry.in", format='aims')
prototype_structures['BaZrS3_P4_mbm']   = ase.io.read("BaZrS3_P4_mbm_geometry.in", format='aims')
def generate_strained_structure(prim, strain_lim):
    strains = np.random.uniform(*strain_lim, (3, ))
    atoms = prim.copy()
    cell_new = prim.cell[:] * (1 + strains)
    atoms.set_cell(cell_new, scale_atoms=True)
    return atoms

def generate_deformed_structure(prim, strain_lim):
    R = np.random.uniform(*strain_lim, (3, 3))
    M = np.eye(3) + R
    atoms = prim.copy()
    cell_new = M @ atoms.cell[:]
    atoms.set_cell(cell_new, scale_atoms=True)
    return atoms

# parameters
strain_lim = [-0.05, 0.05]
n_structures = 30

training_structures = []
for name, prim in prototype_structures.items():
    for it in range(n_structures):
        prim_strained = generate_strained_structure(prim, strain_lim)
        prim_deformed = generate_deformed_structure(prim, strain_lim)

        training_structures.append(prim_strained)
        training_structures.append(prim_deformed)

print('Number of training structures:', len(training_structures))
count = 0
for atoms in training_structures:
    count = count + 1
    ase.io.write('geometry.in-%s'%count,atoms,scaled=True,format='aims')
    print(atoms)
    os.system('grep "atom_frac" geometry.in-%s | wc'%count)

n_structures = 5
rattle_std = 0.04
d_min = 2.4
n_iter = 20

size_vals = {}
size_vals['BaZrS3_Pnma'] = [(2, 2, 2), (3, 3, 3), (2, 2, 3), (2, 3, 2), (3, 2, 2)]
size_vals['BaZrS3_Pm3m'] = [(2, 2, 2), (3, 3, 3), (2, 2, 3), (2, 3, 2), (3, 2, 2)]
size_vals['BaZrS3_I4_mcm'] = [(2, 2, 2), (3, 3, 3), (2, 2, 3), (2, 3, 2), (3, 2, 2)]
size_vals['BaZrS3_P4_mbm'] = [(2, 2, 2), (3, 3, 3), (2, 2, 3), (2, 3, 2), (3, 2, 2)]

training_structures = []
for name, prim in prototype_structures.items():
    for size in size_vals[name]:
        for it in range(n_structures):
            supercell = generate_strained_structure(prim.repeat(size), strain_lim)
            rattled_supercells = generate_mc_rattled_structures(supercell, n_structures=1, rattle_std=rattle_std, d_min=d_min, n_iter=n_iter)
            print(f'{name}, size {size}, natoms {len(supercell)},  volume {supercell.get_volume() / len(supercell):.3f}')
            training_structures.extend(rattled_supercells)

print(training_structures)
print('Number of training structures:', len(training_structures))
for atoms in training_structures:
    count = count + 1
    ase.io.write('geometry.in-%s'%count,atoms,scaled=True,format='aims')
    print(atoms)
    os.system('grep "atom_frac" geometry.in-%s | wc'%count)
