reinitialize
load ../../data/raw_pdb/1BRS.pdb, 1BRS_A_B
hide everything
show cartoon, 1BRS_A_B
color gray80, 1BRS_A_B
set cartoon_transparency, 0.30
set stick_radius, 0.28
select TP_residues, (chain A and resi 108) or (chain A and resi 109) or (chain A and resi 110) or (chain A and resi 8) or (chain B and resi 108) or (chain B and resi 109) or (chain B and resi 110) or (chain B and resi 8)
select FP_residues, (chain A and resi 107) or (chain A and resi 108) or (chain A and resi 109) or (chain A and resi 110) or (chain A and resi 8) or (chain A and resi 96) or (chain A and resi 97) or (chain B and resi 107) or (chain B and resi 108) or (chain B and resi 109) or (chain B and resi 96) or (chain B and resi 97)
select FN_residues, (chain A and resi 107) or (chain A and resi 108) or (chain A and resi 12) or (chain A and resi 15) or (chain A and resi 98) or (chain B and resi 107) or (chain B and resi 108) or (chain B and resi 110) or (chain B and resi 12) or (chain B and resi 98)
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
# png 1BRS_A_B_final_transformerconv_structural_errors_clean.png, dpi=300
