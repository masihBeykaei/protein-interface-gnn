reinitialize
load ../../data/raw_pdb_expanded_bm5/BM5_1A2K_A_B.pdb, BM5_1A2K_A_B
hide everything
show cartoon, BM5_1A2K_A_B
color gray80, BM5_1A2K_A_B
set cartoon_transparency, 0.25
set stick_radius, 0.22
set dash_width, 2.5
set dash_gap, 0.35
select TP_residues, (chain A and resi 123) or (chain A and resi 124) or (chain A and resi 125) or (chain A and resi 126) or (chain A and resi 59) or (chain A and resi 60) or (chain A and resi 95) or (chain B and resi 42) or (chain B and resi 43) or (chain B and resi 67) or (chain B and resi 71) or (chain B and resi 76) or (chain B and resi 78)
select FP_residues, (chain A and resi 298) or (chain A and resi 42) or (chain A and resi 60) or (chain A and resi 61) or (chain A and resi 96) or (chain A and resi 97) or (chain A and resi 98) or (chain B and resi 19) or (chain B and resi 41) or (chain B and resi 42) or (chain B and resi 67) or (chain B and resi 73) or (chain B and resi 74)
select FN_residues, (chain A and resi 120) or (chain A and resi 121) or (chain A and resi 126) or (chain A and resi 42) or (chain A and resi 91) or (chain A and resi 92) or (chain A and resi 94) or (chain A and resi 96) or (chain A and resi 97) or (chain B and resi 39) or (chain B and resi 68) or (chain B and resi 71) or (chain B and resi 76) or (chain B and resi 81)
color green, TP_residues
color red, FP_residues
color orange, FN_residues
show sticks, TP_residues
show sticks, FP_residues
show sticks, FN_residues
distance TP_pair_1, (chain A and resi 123 and name CA), (chain B and resi 42 and name CA)
color green, TP_pair_1
hide labels, TP_pair_1
distance TP_pair_2, (chain A and resi 59 and name CA), (chain B and resi 71 and name CA)
color green, TP_pair_2
hide labels, TP_pair_2
distance TP_pair_3, (chain A and resi 95 and name CA), (chain B and resi 67 and name CA)
color green, TP_pair_3
hide labels, TP_pair_3
distance TP_pair_4, (chain A and resi 125 and name CA), (chain B and resi 78 and name CA)
color green, TP_pair_4
hide labels, TP_pair_4
distance TP_pair_5, (chain A and resi 60 and name CA), (chain B and resi 71 and name CA)
color green, TP_pair_5
hide labels, TP_pair_5
distance TP_pair_6, (chain A and resi 123 and name CA), (chain B and resi 43 and name CA)
color green, TP_pair_6
hide labels, TP_pair_6
distance TP_pair_7, (chain A and resi 124 and name CA), (chain B and resi 43 and name CA)
color green, TP_pair_7
hide labels, TP_pair_7
distance TP_pair_8, (chain A and resi 125 and name CA), (chain B and resi 76 and name CA)
color green, TP_pair_8
hide labels, TP_pair_8
distance TP_pair_9, (chain A and resi 126 and name CA), (chain B and resi 78 and name CA)
color green, TP_pair_9
hide labels, TP_pair_9
distance TP_pair_10, (chain A and resi 124 and name CA), (chain B and resi 42 and name CA)
color green, TP_pair_10
hide labels, TP_pair_10
distance FP_pair_1, (chain A and resi 42 and name CA), (chain B and resi 73 and name CA)
color red, FP_pair_1
hide labels, FP_pair_1
distance FP_pair_2, (chain A and resi 97 and name CA), (chain B and resi 41 and name CA)
color red, FP_pair_2
hide labels, FP_pair_2
distance FP_pair_3, (chain A and resi 96 and name CA), (chain B and resi 67 and name CA)
color red, FP_pair_3
hide labels, FP_pair_3
distance FP_pair_4, (chain A and resi 60 and name CA), (chain B and resi 19 and name CA)
color red, FP_pair_4
hide labels, FP_pair_4
distance FP_pair_5, (chain A and resi 98 and name CA), (chain B and resi 41 and name CA)
color red, FP_pair_5
hide labels, FP_pair_5
distance FP_pair_6, (chain A and resi 61 and name CA), (chain B and resi 19 and name CA)
color red, FP_pair_6
hide labels, FP_pair_6
distance FP_pair_7, (chain A and resi 97 and name CA), (chain B and resi 42 and name CA)
color red, FP_pair_7
hide labels, FP_pair_7
distance FP_pair_8, (chain A and resi 96 and name CA), (chain B and resi 41 and name CA)
color red, FP_pair_8
hide labels, FP_pair_8
distance FP_pair_9, (chain A and resi 298 and name CA), (chain B and resi 42 and name CA)
color red, FP_pair_9
hide labels, FP_pair_9
distance FP_pair_10, (chain A and resi 42 and name CA), (chain B and resi 74 and name CA)
color red, FP_pair_10
hide labels, FP_pair_10
distance FN_pair_1, (chain A and resi 94 and name CA), (chain B and resi 71 and name CA)
color orange, FN_pair_1
hide labels, FN_pair_1
distance FN_pair_2, (chain A and resi 120 and name CA), (chain B and resi 76 and name CA)
color orange, FN_pair_2
hide labels, FN_pair_2
distance FN_pair_3, (chain A and resi 121 and name CA), (chain B and resi 76 and name CA)
color orange, FN_pair_3
hide labels, FN_pair_3
distance FN_pair_4, (chain A and resi 97 and name CA), (chain B and resi 39 and name CA)
color orange, FN_pair_4
hide labels, FN_pair_4
distance FN_pair_5, (chain A and resi 92 and name CA), (chain B and resi 71 and name CA)
color orange, FN_pair_5
hide labels, FN_pair_5
distance FN_pair_6, (chain A and resi 42 and name CA), (chain B and resi 76 and name CA)
color orange, FN_pair_6
hide labels, FN_pair_6
distance FN_pair_7, (chain A and resi 126 and name CA), (chain B and resi 81 and name CA)
color orange, FN_pair_7
hide labels, FN_pair_7
distance FN_pair_8, (chain A and resi 91 and name CA), (chain B and resi 71 and name CA)
color orange, FN_pair_8
hide labels, FN_pair_8
distance FN_pair_9, (chain A and resi 96 and name CA), (chain B and resi 39 and name CA)
color orange, FN_pair_9
hide labels, FN_pair_9
distance FN_pair_10, (chain A and resi 97 and name CA), (chain B and resi 68 and name CA)
color orange, FN_pair_10
hide labels, FN_pair_10
zoom TP_residues or FP_residues or FN_residues
orient
bg_color white
# Optional high-resolution export:
# ray 1800, 1400
# png BM5_1A2K_A_B_final_transformerconv_structural_errors.png, dpi=300
