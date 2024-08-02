import numpy as np

# 元素記号と原子番号の対応表
element_to_atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
    'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
    'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84,
    'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94,
    'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
    'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def read_xyz(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    num_atoms = int(lines[0])
    comment = lines[1].strip()
    atoms = []
    coordinates = []
    
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coordinates.append([float(x) for x in parts[1:4]])
    
    return atoms, np.array(coordinates), comment

def write_xyz(filename, atoms, coordinates, comment=''):
    num_atoms = len(atoms)
    with open(filename, 'w') as file:
        file.write("\n\n")
        file.write(f"{num_atoms}\n")
        file.write("\n\n")
        file.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atoms, coordinates):
            atomic_number = element_to_atomic_number.get(atom, 0)
            file.write(f"{atomic_number} {0.000} {x:.6f} {y:.6f} {z:.6f}\n")

def transform_coordinates(atoms, coordinates, N_O, N_X, N_XZ):
    origin = coordinates[N_O]
    x_atom = coordinates[N_X]
    xz_atom = coordinates[N_XZ]

    # Translate to origin
    new_coords = coordinates - origin

    # Create new x-axis
    x_axis = (x_atom - origin) / np.linalg.norm(x_atom - origin)
    
    # Create new z-axis
    temp_z = (xz_atom - origin) / np.linalg.norm(xz_atom - origin)
    z_axis = temp_z - np.dot(temp_z, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    
    # Create new y-axis
    y_axis = np.cross(z_axis, x_axis)

    # Transformation matrix
    transform_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Apply transformation
    new_coords = np.dot(new_coords, transform_matrix)
    
    return new_coords/0.52917720859
def transform_coordinates4(atoms, coordinates, N_O, N_X, N_XZ1, N_XZ2):
    origin = coordinates[N_O]
    x_atom = coordinates[N_X]
    xz1_atom = coordinates[N_XZ1]
    xz2_atom = coordinates[N_XZ2]

    # Translate to origin
    new_coords = coordinates - origin

    # Create new x-axis
    x_axis = (x_atom - origin) / np.linalg.norm(x_atom - origin)
    
    # Determine y_axis using xz1_atom and xz2_atom
    midpoint = (xz1_atom + xz2_atom) / 2
    temp_y = (midpoint - origin) - np.dot((midpoint - origin), x_axis) * x_axis
    temp_y /= np.linalg.norm(temp_y)
    
    # Calculate the z component difference and ensure xz1_atom has a greater z component
    if np.dot((xz1_atom - origin), temp_y) < np.dot((xz2_atom - origin), temp_y):
        xz1_atom, xz2_atom = xz2_atom, xz1_atom

    # Recalculate y_axis
    z_axis = (xz1_atom - xz2_atom)
    z_axis -= np.dot(z_axis, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    
    # Create new z-axis
    y_axis = np.cross(x_axis, z_axis)

    # Transformation matrix
    transform_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Apply transformation
    new_coords = np.dot(new_coords, transform_matrix)
    
    return new_coords / 0.52917720859
# Example usage
input_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/TS1R_B3LYP_6-31Gd_PCM_TS.xyz"
output_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/TS1R_B3LYP_6-31Gd_PCM_TS_new.xyz"
N_O = 43-1
N_X = 42-1
N_XZ = 59-1
N_XZ2=44-1

atoms, coordinates, comment = read_xyz(input_filename)
new_coordinates = transform_coordinates(atoms, coordinates, N_O, N_X, N_XZ)
write_xyz(output_filename, atoms, new_coordinates, comment)

output_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/TS1R_B3LYP_6-31Gd_PCM_TS_new_4.xyz"
new_coordinates = transform_coordinates4(atoms, coordinates, N_O, N_X, N_XZ,N_XZ2)
write_xyz(output_filename, atoms, new_coordinates, comment)

# Example usage
input_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/dip_acetophenone_UFF_B3LYP_631Gd_PCM_IRC.xyz"
output_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/dip_acetophenone_UFF_B3LYP_631Gd_PCM_IRC_new.xyz"
N_O = 29-1
N_X = 31-1
N_XZ = 30-1
N_XZ2=27-1

atoms, coordinates, comment = read_xyz(input_filename)
new_coordinates = transform_coordinates(atoms, coordinates, N_O, N_X, N_XZ)
write_xyz(output_filename, atoms, new_coordinates, comment)

output_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/dip_acetophenone_UFF_B3LYP_631Gd_PCM_IRC_new_4.xyz"
new_coordinates = transform_coordinates4(atoms, coordinates, N_O, N_X, N_XZ,N_XZ2)
write_xyz(output_filename, atoms, new_coordinates, comment)

# Example usage
input_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/Ru_acetophenone_TS_TS.xyz"
output_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/Ru_acetophenone_TS_TS_new.xyz"
N_O = 2-1
N_X = 3-1
N_XZ = 1-1
N_XZ2=4-1

atoms, coordinates, comment = read_xyz(input_filename)
new_coordinates = transform_coordinates(atoms, coordinates, N_O, N_X, N_XZ)
write_xyz(output_filename, atoms, new_coordinates, comment)

output_filename = "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/Ru_acetophenone_TS_TS_new_4.xyz"
new_coordinates = transform_coordinates4(atoms, coordinates, N_O, N_X, N_XZ,N_XZ2)
write_xyz(output_filename, atoms, new_coordinates, comment)