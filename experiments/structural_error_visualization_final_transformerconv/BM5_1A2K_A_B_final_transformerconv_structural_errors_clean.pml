reinitialize
load ../../data/raw_pdb_expanded_bm5/BM5_1A2K_A_B.pdb, BM5_1A2K_A_B
hide everything
show cartoon, BM5_1A2K_A_B
color gray80, BM5_1A2K_A_B
set cartoon_transparency, 0.30
set stick_radius, 0.28
select TP_residues, (chain A and resi 123) or (chain A and resi 124) or (chain A and resi 125) or (chain A and resi 126) or (chain A and resi 59) or (chain A and resi 60) or (chain A and resi 95) or (chain B and resi 42) or (chain B and resi 43) or (chain B and resi 67) or (chain B and resi 71) or (chain B and resi 76) or (chain B and resi 78)
select FP_residues, (chain A and resi 298) or (chain A and resi 42) or (chain A and resi 60) or (chain A and resi 61) or (chain A and resi 96) or (chain A and resi 97) or (chain A and resi 98) or (chain B and resi 19) or (chain B and resi 41) or (chain B and resi 42) or (chain B and resi 67) or (chain B and resi 73) or (chain B and resi 74)
select FN_residues, (chain A and resi 120) or (chain A and resi 121) or (chain A and resi 126) or (chain A and resi 42) or (chain A and resi 91) or (chain A and resi 92) or (chain A and resi 94) or (chain A and resi 96) or (chain A and resi 97) or (chain B and resi 39) or (chain B and resi 68) or (chain B and resi 71) or (chain B and resi 76) or (chain B and resi 81)
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
# png BM5_1A2K_A_B_final_transformerconv_structural_errors_clean.png, dpi=300
