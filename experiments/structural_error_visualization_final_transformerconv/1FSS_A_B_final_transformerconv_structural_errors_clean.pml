reinitialize
load ../../data/raw_pdb/1FSS.pdb, 1FSS_A_B
hide everything
show cartoon, 1FSS_A_B
color gray80, 1FSS_A_B
set cartoon_transparency, 0.30
set stick_radius, 0.28
select TP_residues, (chain A and resi 276) or (chain A and resi 285) or (chain A and resi 334) or (chain A and resi 335) or (chain A and resi 70) or (chain A and resi 71) or (chain B and resi 30) or (chain B and resi 31) or (chain B and resi 32) or (chain B and resi 33) or (chain B and resi 8) or (chain B and resi 9)
select FP_residues, (chain A and resi 285) or (chain A and resi 287) or (chain A and resi 335) or (chain A and resi 68) or (chain A and resi 70) or (chain A and resi 72) or (chain B and resi 11) or (chain B and resi 29) or (chain B and resi 30) or (chain B and resi 31) or (chain B and resi 34) or (chain B and resi 8) or (chain B and resi 9)
select FN_residues, (chain A and resi 282) or (chain A and resi 284) or (chain A and resi 70) or (chain A and resi 74) or (chain A and resi 76) or (chain A and resi 78) or (chain A and resi 86) or (chain A and resi 87) or (chain B and resi 11) or (chain B and resi 27) or (chain B and resi 32) or (chain B and resi 33) or (chain B and resi 34) or (chain B and resi 37) or (chain B and resi 61)
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
# png 1FSS_A_B_final_transformerconv_structural_errors_clean.png, dpi=300
