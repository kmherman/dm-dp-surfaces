import numpy as np
from scipy.interpolate import interpn


def get_dp(theta, R0, dR, water_name, path_to_surfs='H2O/hf_avtz',
           method='linear', bohr=False, radians=False):
    """
    Linearly interpolate distributed polarizability surfaces to
    estimate the polarizabilities for a specified intramolecular
    geometry.

    args:
    theta (float): H-O-H angle in degrees
    R0 (float): Average R_OH value in angstrom
    dR (float): Difference between R_OH values in angstrom
    water_name (str): Identifying string for water molecule, typically w{n}
    return: string of distributed polarizabilities for the O, H1, H2 atoms
    """
    if bohr:
        R0 /= 1.88973
        dR /= 1.88973

    if radians:
        theta *= (180/np.Pi)

    theta_range = np.arange(60, 150, 5)  # define grid of points
    R0_range = np.arange(0.81, 1.26, 0.05)
    dR_range = np.arange(0.00, 0.44, 0.04)

    dp_o = np.zeros((8, 8))
    dp_h1 = np.zeros((8, 8))
    dp_h2 = np.zeros((8, 8))

    atom_types = ['O', 'H1', 'H2']
    pol = ['10_10', '10_11c', '10_20', '10_21c', '10_22c', '11c_11c', '11c_20',
           '11c_21c', '11c_22c', '11s_11s', '11s_21s', '11s_22s', '20_20',
           '20_21c', '20_22c', '21c_21c', '21c_22c', '21s_21s', '21s_22s',
           '22c_22c', '22s_22s']
    pol_pos = [[0, 0], [0, 1], [0, 3], [0, 4], [0, 6], [1, 1], [1, 3],
               [1, 4], [1, 6], [2, 2], [2, 5], [2, 7], [3, 3], [3, 4],
               [3, 6], [4, 4], [4, 6], [5, 5], [5, 7], [6, 6], [7, 7]]

    for i in range(len(atom_types)):
        for j in range(len(pol)):
            with open(f'{path_to_surfs}/{atom_types[i]}_p{pol[j]}.npy', 'rb') as f:
                p = np.load(f)
                p_interp = interpn((theta_range, R0_range, dR_range), p,
                                   [theta, R0, dR], method=method)
                if atom_types[i] == 'O':
                    dp_o[pol_pos[j][0], pol_pos[j][1]] = p_interp
                    dp_o[pol_pos[j][1], pol_pos[j][0]] = p_interp
                elif atom_types[i] == 'H1':
                    dp_h1[pol_pos[j][0], pol_pos[j][1]] = p_interp
                    dp_h1[pol_pos[j][1], pol_pos[j][0]] = p_interp
                else:
                    dp_h2[pol_pos[j][0], pol_pos[j][1]] = p_interp
                    dp_h2[pol_pos[j][1], pol_pos[j][0]] = p_interp

    dp_o_string = f'''
    ALPHA  {water_name}  SITE-NAMES  O  O  RANK 1 TO 2 INDEX   1 FREQSQ       0.0000000
    {dp_o[0,0]:.5f} {dp_o[0,1]:.5f} 0.0 {dp_o[0,3]:.5f} {dp_o[0,4]:.5f} 0.0 {dp_o[0,6]:.5f} 0.0
    {dp_o[1,0]:.5f} {dp_o[1,1]:.5f} 0.0 {dp_o[1,3]:.5f} {dp_o[1,4]:.5f} 0.0 {dp_o[1,6]:.5f} 0.0
    0.0 0.0 {dp_o[2,2]:.5f} 0.0 0.0 {dp_o[2,5]:.5f} 0.0 {dp_o[2,7]:.5f}
    {dp_o[3,0]:.5f} {dp_o[3,1]:.5f} 0.0 {dp_o[3,3]:.5f} {dp_o[3,4]:.5f} 0.0 {dp_o[3,6]:.5f} 0.0
    {dp_o[4,0]:.5f} {dp_o[4,1]:.5f} 0.0 {dp_o[4,3]:.5f} {dp_o[4,4]:.5f} 0.0 {dp_o[4,6]:.5f} 0.0
    0.0 0.0 {dp_o[5,2]:.5f} 0.0 0.0 {dp_o[5,5]:.5f} 0.0 {dp_o[5,7]:.5f}
    {dp_o[6,0]:.5f} {dp_o[6,1]:.5f} 0.0 {dp_o[6,3]:.5f} {dp_o[6,4]:.5f} 0.0 {dp_o[6,6]:.5f} 0.0
    0.0 0.0 {dp_o[7,2]:.5f} 0.0 0.0 {dp_o[7,5]:.5f} 0.0 {dp_o[7,7]:.5f}'''

    dp_h1_string = f'''
    ALPHA  {water_name}  SITE-NAMES  H1  H1  RANK 1 TO 2 INDEX   1 FREQSQ       0.0000000
    {dp_h1[0,0]:.5f} {dp_h1[0,1]:.5f} 0.0 {dp_h1[0,3]:.5f} {dp_h1[0,4]:.5f} 0.0 {dp_h1[0,6]:.5f} 0.0
    {dp_h1[1,0]:.5f} {dp_h1[1,1]:.5f} 0.0 {dp_h1[1,3]:.5f} {dp_h1[1,4]:.5f} 0.0 {dp_h1[1,6]:.5f} 0.0
    0.0 0.0 {dp_h1[2,2]:.5f} 0.0 0.0 {dp_h1[2,5]:.5f} 0.0 {dp_h1[2,7]:.5f}
    {dp_h1[3,0]:.5f} {dp_h1[3,1]:.5f} 0.0 {dp_h1[3,3]:.5f} {dp_h1[3,4]:.5f} 0.0 {dp_h1[3,6]:.5f} 0.0
    {dp_h1[4,0]:.5f} {dp_h1[4,1]:.5f} 0.0 {dp_h1[4,3]:.5f} {dp_h1[4,4]:.5f} 0.0 {dp_h1[4,6]:.5f} 0.0
    0.0 0.0 {dp_h1[5,2]:.5f} 0.0 0.0 {dp_h1[5,5]:.5f} 0.0 {dp_h1[5,7]:.5f}
    {dp_h1[6,0]:.5f} {dp_h1[6,1]:.5f} 0.0 {dp_h1[6,3]:.5f} {dp_h1[6,4]:.5f} 0.0 {dp_h1[6,6]:.5f} 0.0
    0.0 0.0 {dp_h1[7,2]:.5f} 0.0 0.0 {dp_h1[7,5]:.5f} 0.0 {dp_h1[7,7]:.5f}'''

    dp_h2_string = f'''
    ALPHA  {water_name}  SITE-NAMES  H2  H2  RANK 1 TO 2 INDEX   1 FREQSQ       0.0000000
    {dp_h2[0,0]:.5f} {dp_h2[0,1]:.5f} 0.0 {dp_h2[0,3]:.5f} {dp_h2[0,4]:.5f} 0.0 {dp_h2[0,6]:.5f} 0.0
    {dp_h2[1,0]:.5f} {dp_h2[1,1]:.5f} 0.0 {dp_h2[1,3]:.5f} {dp_h2[1,4]:.5f} 0.0 {dp_h2[1,6]:.5f} 0.0
    0.0 0.0 {dp_h2[2,2]:.5f} 0.0 0.0 {dp_h2[2,5]:.5f} 0.0 {dp_h2[2,7]:.5f}
    {dp_h2[3,0]:.5f} {dp_h2[3,1]:.5f} 0.0 {dp_h2[3,3]:.5f} {dp_h2[3,4]:.5f} 0.0 {dp_h2[3,6]:.5f} 0.0
    {dp_h2[4,0]:.5f} {dp_h2[4,1]:.5f} 0.0 {dp_h2[4,3]:.5f} {dp_h2[4,4]:.5f} 0.0 {dp_h2[4,6]:.5f} 0.0
    0.0 0.0 {dp_h2[5,2]:.5f} 0.0 0.0 {dp_h2[5,5]:.5f} 0.0 {dp_h2[5,7]:.5f}
    {dp_h2[6,0]:.5f} {dp_h2[6,1]:.5f} 0.0 {dp_h2[6,3]:.5f} {dp_h2[6,4]:.5f} 0.0 {dp_h2[6,6]:.5f} 0.0
    0.0 0.0 {dp_h2[7,2]:.5f} 0.0 0.0 {dp_h2[7,5]:.5f} 0.0 {dp_h2[7,7]:.5f}'''

    return dp_o_string, dp_h1_string, dp_h2_string


def get_dm(theta, R0, dR, path_to_surfs='H2O/hf_avtz', method='linear',
           bohr=False, radians=False):
    """
    Linearly interpolate distributed multipole surfaces to estimate the
    multipoles for a specified intramolecular geometry.
    args:
    theta (float): H-O-H angle in degrees
    R0 (float): Average R_OH value in angstrom
    dR (float): Difference between R_OH values in angstrom
    water_name (str): Identifying string for water molecule, typically w_{n}
    return: string of distributed multipoles for the O, H1, H2 atoms
    """
    if bohr:
        R0 /= 1.88973
        dR /= 1.88973

    if radians:
        theta *= (180/np.Pi)

    theta_range = np.arange(60, 150, 5)  # define grid of points
    R0_range = np.arange(0.81, 1.26, 0.05)
    dR_range = np.arange(0.00, 0.44, 0.04)

    dm_o = np.zeros((5, 5))
    dm_h1 = np.zeros((5, 5))
    dm_h2 = np.zeros((5, 5))

    atom_types = ['O', 'H1', 'H2']
    multipoles = ['q00', 'q10', 'q11c', 'q20', 'q21c', 'q22c', 'q30', 'q31c',
                  'q32c', 'q33c', 'q40', 'q41c', 'q42c', 'q43c', 'q44c']
    multi_pos = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0],
                 [3, 1], [3, 2], [3, 3], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]

    for i in range(len(atom_types)):
        for j in range(len(multipoles)):
            with open(f'{path_to_surfs}/{atom_types[i]}_{multipoles[j]}.npy', 'rb') as f:
                q = np.load(f)
                q_interp = interpn((theta_range, R0_range, dR_range), q,
                                   [theta, R0, dR], method=method)
                if atom_types[i] == 'O':
                    dm_o[multi_pos[j][0], multi_pos[j][1]] = q_interp
                elif atom_types[i] == 'H1':
                    dm_h1[multi_pos[j][0], multi_pos[j][1]] = q_interp
                else:
                    dm_h2[multi_pos[j][0], multi_pos[j][1]] = q_interp

    dm_o_string = f'''
    Q00  =   {dm_o[0,0]:.5f}
    Q10  =   {dm_o[1,0]:.5f}  Q11c =   {dm_o[1,1]:.5f}
    Q20  =   {dm_o[2,0]:.5f}  Q21c =   {dm_o[2,1]:.5f}  Q22c =   {dm_o[2,2]:.5f}
    Q30  =   {dm_o[3,0]:.5f}  Q31c =   {dm_o[3,1]:.5f}  Q32c =   {dm_o[3,2]:.5f}  Q33c =  {dm_o[3,3]:.5f}
    Q40  =   {dm_o[4,0]:.5f}  Q41c =   {dm_o[4,1]:.5f}  Q42c =   {dm_o[4,2]:.5f}  Q43c =  {dm_o[4,3]:.5f} Q44c =   {dm_o[4,4]:.5f}'''

    dm_h1_string = f'''
    Q00  =   {dm_h1[0,0]:.5f}
    Q10  =   {dm_h1[1,0]:.5f}  Q11c =   {dm_h1[1,1]:.5f}
    Q20  =   {dm_h1[2,0]:.5f}  Q21c =   {dm_h1[2,1]:.5f}  Q22c =   {dm_h1[2,2]:.5f}
    Q30  =   {dm_h1[3,0]:.5f}  Q31c =   {dm_h1[3,1]:.5f}  Q32c =   {dm_h1[3,2]:.5f}  Q33c =  {dm_h1[3,3]:.5f}
    Q40  =   {dm_h1[4,0]:.5f}  Q41c =   {dm_h1[4,1]:.5f}  Q42c =   {dm_h1[4,2]:.5f}  Q43c =  {dm_h1[4,3]:.5f} Q44c =   {dm_h1[4,4]:.5f}'''

    dm_h2_string = f'''
    Q00  =   {dm_h2[0,0]:.5f}
    Q10  =   {dm_h2[1,0]:.5f}  Q11c =   {dm_h2[1,1]:.5f}
    Q20  =   {dm_h2[2,0]:.5f}  Q21c =   {dm_h2[2,1]:.5f}  Q22c =   {dm_h2[2,2]:.5f}
    Q30  =   {dm_h2[3,0]:.5f}  Q31c =   {dm_h2[3,1]:.5f}  Q32c =   {dm_h2[3,2]:.5f}  Q33c =  {dm_h2[3,3]:.5f}
    Q40  =   {dm_h2[4,0]:.5f}  Q41c =   {dm_h2[4,1]:.5f}  Q42c =   {dm_h2[4,2]:.5f}  Q43c =  {dm_h2[4,3]:.5f} Q44c =   {dm_h2[4,4]:.5f}'''

    return dm_o_string, dm_h1_string, dm_h2_string
