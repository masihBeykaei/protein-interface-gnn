reinitialize
load ../../data/raw_pdb/3HMX.pdb, 3HMX_LH_AB
hide everything
show cartoon, 3HMX_LH_AB
color gray80, 3HMX_LH_AB
set cartoon_transparency, 0.25
set stick_radius, 0.22
set dash_width, 2.5
set dash_gap, 0.35
select TP_residues, (chain A and resi 18) or (chain A and resi 20) or (chain A and resi 55) or (chain A and resi 56) or (chain A and resi 61) or (chain A and resi 62) or (chain H and resi 101) or (chain H and resi 102) or (chain H and resi 103) or (chain H and resi 33) or (chain L and resi 92) or (chain L and resi 93) or (chain L and resi 94)
select FP_residues, (chain A and resi 18) or (chain A and resi 19) or (chain A and resi 20) or (chain A and resi 43) or (chain A and resi 56) or (chain A and resi 57) or (chain A and resi 58) or (chain A and resi 61) or (chain H and resi 102) or (chain H and resi 103) or (chain H and resi 104) or (chain H and resi 27) or (chain H and resi 32) or (chain H and resi 52) or (chain L and resi 91) or (chain L and resi 94) or (chain L and resi 95)
select FN_residues, (chain A and resi 15) or (chain A and resi 45) or (chain A and resi 47) or (chain A and resi 56) or (chain A and resi 58) or (chain A and resi 59) or (chain H and resi 107) or (chain H and resi 2) or (chain H and resi 26) or (chain H and resi 32) or (chain H and resi 57) or (chain H and resi 59) or (chain H and resi 98) or (chain L and resi 94) or (chain L and resi 96)
color green, TP_residues
color red, FP_residues
color orange, FN_residues
show sticks, TP_residues
show sticks, FP_residues
show sticks, FN_residues
distance TP_pair_1, (chain H and resi 101 and name CA), (chain A and resi 62 and name CA)
color green, TP_pair_1
hide labels, TP_pair_1
distance TP_pair_2, (chain H and resi 103 and name CA), (chain A and resi 56 and name CA)
color green, TP_pair_2
hide labels, TP_pair_2
distance TP_pair_3, (chain L and resi 94 and name CA), (chain A and resi 18 and name CA)
color green, TP_pair_3
hide labels, TP_pair_3
distance TP_pair_4, (chain L and resi 92 and name CA), (chain A and resi 20 and name CA)
color green, TP_pair_4
hide labels, TP_pair_4
distance TP_pair_5, (chain L and resi 93 and name CA), (chain A and resi 20 and name CA)
color green, TP_pair_5
hide labels, TP_pair_5
distance TP_pair_6, (chain H and resi 33 and name CA), (chain A and resi 61 and name CA)
color green, TP_pair_6
hide labels, TP_pair_6
distance TP_pair_7, (chain H and resi 102 and name CA), (chain A and resi 56 and name CA)
color green, TP_pair_7
hide labels, TP_pair_7
distance TP_pair_8, (chain H and resi 102 and name CA), (chain A and resi 62 and name CA)
color green, TP_pair_8
hide labels, TP_pair_8
distance TP_pair_9, (chain L and resi 93 and name CA), (chain A and resi 18 and name CA)
color green, TP_pair_9
hide labels, TP_pair_9
distance TP_pair_10, (chain H and resi 102 and name CA), (chain A and resi 55 and name CA)
color green, TP_pair_10
hide labels, TP_pair_10
distance FP_pair_1, (chain H and resi 32 and name CA), (chain A and resi 61 and name CA)
color red, FP_pair_1
hide labels, FP_pair_1
distance FP_pair_2, (chain H and resi 102 and name CA), (chain A and resi 57 and name CA)
color red, FP_pair_2
hide labels, FP_pair_2
distance FP_pair_3, (chain H and resi 52 and name CA), (chain A and resi 61 and name CA)
color red, FP_pair_3
hide labels, FP_pair_3
distance FP_pair_4, (chain H and resi 103 and name CA), (chain A and resi 57 and name CA)
color red, FP_pair_4
hide labels, FP_pair_4
distance FP_pair_5, (chain L and resi 94 and name CA), (chain A and resi 19 and name CA)
color red, FP_pair_5
hide labels, FP_pair_5
distance FP_pair_6, (chain L and resi 91 and name CA), (chain A and resi 20 and name CA)
color red, FP_pair_6
hide labels, FP_pair_6
distance FP_pair_7, (chain L and resi 95 and name CA), (chain A and resi 18 and name CA)
color red, FP_pair_7
hide labels, FP_pair_7
distance FP_pair_8, (chain H and resi 104 and name CA), (chain A and resi 56 and name CA)
color red, FP_pair_8
hide labels, FP_pair_8
distance FP_pair_9, (chain H and resi 27 and name CA), (chain A and resi 43 and name CA)
color red, FP_pair_9
hide labels, FP_pair_9
distance FP_pair_10, (chain L and resi 91 and name CA), (chain A and resi 58 and name CA)
color red, FP_pair_10
hide labels, FP_pair_10
distance FN_pair_1, (chain L and resi 96 and name CA), (chain A and resi 56 and name CA)
color orange, FN_pair_1
hide labels, FN_pair_1
distance FN_pair_2, (chain H and resi 57 and name CA), (chain A and resi 15 and name CA)
color orange, FN_pair_2
hide labels, FN_pair_2
distance FN_pair_3, (chain L and resi 94 and name CA), (chain A and resi 59 and name CA)
color orange, FN_pair_3
hide labels, FN_pair_3
distance FN_pair_4, (chain H and resi 26 and name CA), (chain A and resi 45 and name CA)
color orange, FN_pair_4
hide labels, FN_pair_4
distance FN_pair_5, (chain H and resi 98 and name CA), (chain A and resi 45 and name CA)
color orange, FN_pair_5
hide labels, FN_pair_5
distance FN_pair_6, (chain H and resi 32 and name CA), (chain A and resi 47 and name CA)
color orange, FN_pair_6
hide labels, FN_pair_6
distance FN_pair_7, (chain L and resi 96 and name CA), (chain A and resi 58 and name CA)
color orange, FN_pair_7
hide labels, FN_pair_7
distance FN_pair_8, (chain H and resi 107 and name CA), (chain A and resi 45 and name CA)
color orange, FN_pair_8
hide labels, FN_pair_8
distance FN_pair_9, (chain H and resi 2 and name CA), (chain A and resi 45 and name CA)
color orange, FN_pair_9
hide labels, FN_pair_9
distance FN_pair_10, (chain H and resi 59 and name CA), (chain A and resi 15 and name CA)
color orange, FN_pair_10
hide labels, FN_pair_10
zoom TP_residues or FP_residues or FN_residues
orient
bg_color white
# Optional high-resolution export:
# ray 1800, 1400
# png 3HMX_LH_AB_final_transformerconv_structural_errors.png, dpi=300
