reinitialize
load ../../data/raw_pdb/3HMX.pdb, 3HMX_LH_AB
hide everything
show cartoon, 3HMX_LH_AB
color gray80, 3HMX_LH_AB
set cartoon_transparency, 0.30
set stick_radius, 0.28
select TP_residues, (chain A and resi 18) or (chain A and resi 20) or (chain A and resi 55) or (chain A and resi 56) or (chain A and resi 61) or (chain A and resi 62) or (chain H and resi 101) or (chain H and resi 102) or (chain H and resi 103) or (chain H and resi 33) or (chain L and resi 92) or (chain L and resi 93) or (chain L and resi 94)
select FP_residues, (chain A and resi 18) or (chain A and resi 19) or (chain A and resi 20) or (chain A and resi 43) or (chain A and resi 56) or (chain A and resi 57) or (chain A and resi 58) or (chain A and resi 61) or (chain H and resi 102) or (chain H and resi 103) or (chain H and resi 104) or (chain H and resi 27) or (chain H and resi 32) or (chain H and resi 52) or (chain L and resi 91) or (chain L and resi 94) or (chain L and resi 95)
select FN_residues, (chain A and resi 15) or (chain A and resi 45) or (chain A and resi 47) or (chain A and resi 56) or (chain A and resi 58) or (chain A and resi 59) or (chain H and resi 107) or (chain H and resi 2) or (chain H and resi 26) or (chain H and resi 32) or (chain H and resi 57) or (chain H and resi 59) or (chain H and resi 98) or (chain L and resi 94) or (chain L and resi 96)
show sticks, TP_residues
show sticks, FP_residues
show sticks, FN_residues
color green, TP_residues
color red, FP_residues
color orange, FN_residues
zoom TP_residues or FP_residues or FN_residues
orient
bg_color white
# Optional high-resolution export:
# ray 1800, 1400
# png 3HMX_LH_AB_final_transformerconv_structural_errors_clean.png, dpi=300
