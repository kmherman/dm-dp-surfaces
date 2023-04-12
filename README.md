# dm-dp-surfaces

This repo contains geometry-dependent distributed multipoles (up to hexadecapole) and distributed polarizability surfaces (up to quadrupole-quadrupole) for water derived from either the HF/aug-cc-pVTZ (H2O/hf_avtz) or PBE0/aug-cc-pVTZ (H2O/dft_avtz) level of theory.

Interpolate_dm_dp.py contains functions that will perform a linear interpolation of the distributed multipole and distributed polarizability surfaces given an intramolecular geometry of water specified by the H-O-H angle, the average R_OH distance, and the difference between R_OH values. 
