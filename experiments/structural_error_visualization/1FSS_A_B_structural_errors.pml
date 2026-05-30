# PyMOL structural error visualization for 1FSS_A_B
# Green = TP, Red = FP, Orange = FN

reinitialize
load ../../data/raw_pdb/1FSS.pdb, 1FSS_A_B

hide everything
show cartoon
color gray80
set cartoon_transparency, 0.25
set stick_radius, 0.22
set dash_width, 2.5
set dash_gap, 0.35

select TP_residues, (chain A and resi 334) or (chain A and resi 335) or (chain A and resi 336) or (chain A and resi 73) or (chain A and resi 76) or (chain B and resi 10) or (chain B and resi 11) or (chain B and resi 12) or (chain B and resi 30) or (chain B and resi 31) or (chain B and resi 32) or (chain B and resi 6)
select FP_residues, (chain A and resi 331) or (chain A and resi 332) or (chain A and resi 333) or (chain A and resi 334) or (chain A and resi 335) or (chain B and resi 29) or (chain B and resi 30) or (chain B and resi 31)
select FN_residues, (chain A and resi 276) or (chain A and resi 279) or (chain A and resi 285) or (chain A and resi 70) or (chain A and resi 74) or (chain A and resi 87) or (chain A and resi 90) or (chain B and resi 11) or (chain B and resi 27) or (chain B and resi 32) or (chain B and resi 33) or (chain B and resi 34) or (chain B and resi 35) or (chain B and resi 9)

color green, TP_residues
color red, FP_residues
color orange, FN_residues

show sticks, TP_residues
show sticks, FP_residues
show sticks, FN_residues

distance TP_pair_1, (chain A and resi 334 and name CA), (chain B and resi 31 and name CA)
color green, TP_pair_1
hide labels, TP_pair_1
distance TP_pair_2, (chain A and resi 335 and name CA), (chain B and resi 30 and name CA)
color green, TP_pair_2
hide labels, TP_pair_2
distance TP_pair_3, (chain A and resi 334 and name CA), (chain B and resi 30 and name CA)
color green, TP_pair_3
hide labels, TP_pair_3
distance TP_pair_4, (chain A and resi 73 and name CA), (chain B and resi 11 and name CA)
color green, TP_pair_4
hide labels, TP_pair_4
distance TP_pair_5, (chain A and resi 335 and name CA), (chain B and resi 32 and name CA)
color green, TP_pair_5
hide labels, TP_pair_5
distance TP_pair_6, (chain A and resi 334 and name CA), (chain B and resi 32 and name CA)
color green, TP_pair_6
hide labels, TP_pair_6
distance TP_pair_7, (chain A and resi 73 and name CA), (chain B and resi 10 and name CA)
color green, TP_pair_7
hide labels, TP_pair_7
distance TP_pair_8, (chain A and resi 336 and name CA), (chain B and resi 30 and name CA)
color green, TP_pair_8
hide labels, TP_pair_8
distance TP_pair_9, (chain A and resi 73 and name CA), (chain B and resi 6 and name CA)
color green, TP_pair_9
hide labels, TP_pair_9
distance TP_pair_10, (chain A and resi 76 and name CA), (chain B and resi 12 and name CA)
color green, TP_pair_10
hide labels, TP_pair_10
distance FP_pair_1, (chain A and resi 333 and name CA), (chain B and resi 31 and name CA)
color red, FP_pair_1
hide labels, FP_pair_1
distance FP_pair_2, (chain A and resi 335 and name CA), (chain B and resi 31 and name CA)
color red, FP_pair_2
hide labels, FP_pair_2
distance FP_pair_3, (chain A and resi 333 and name CA), (chain B and resi 29 and name CA)
color red, FP_pair_3
hide labels, FP_pair_3
distance FP_pair_4, (chain A and resi 333 and name CA), (chain B and resi 30 and name CA)
color red, FP_pair_4
hide labels, FP_pair_4
distance FP_pair_5, (chain A and resi 332 and name CA), (chain B and resi 31 and name CA)
color red, FP_pair_5
hide labels, FP_pair_5
distance FP_pair_6, (chain A and resi 334 and name CA), (chain B and resi 29 and name CA)
color red, FP_pair_6
hide labels, FP_pair_6
distance FP_pair_7, (chain A and resi 335 and name CA), (chain B and resi 29 and name CA)
color red, FP_pair_7
hide labels, FP_pair_7
distance FP_pair_8, (chain A and resi 332 and name CA), (chain B and resi 29 and name CA)
color red, FP_pair_8
hide labels, FP_pair_8
distance FP_pair_9, (chain A and resi 332 and name CA), (chain B and resi 30 and name CA)
color red, FP_pair_9
hide labels, FP_pair_9
distance FP_pair_10, (chain A and resi 331 and name CA), (chain B and resi 31 and name CA)
color red, FP_pair_10
hide labels, FP_pair_10
distance FN_pair_1, (chain A and resi 70 and name CA), (chain B and resi 33 and name CA)
color orange, FN_pair_1
hide labels, FN_pair_1
distance FN_pair_2, (chain A and resi 74 and name CA), (chain B and resi 32 and name CA)
color orange, FN_pair_2
hide labels, FN_pair_2
distance FN_pair_3, (chain A and resi 74 and name CA), (chain B and resi 33 and name CA)
color orange, FN_pair_3
hide labels, FN_pair_3
distance FN_pair_4, (chain A and resi 70 and name CA), (chain B and resi 34 and name CA)
color orange, FN_pair_4
hide labels, FN_pair_4
distance FN_pair_5, (chain A and resi 279 and name CA), (chain B and resi 35 and name CA)
color orange, FN_pair_5
hide labels, FN_pair_5
distance FN_pair_6, (chain A and resi 87 and name CA), (chain B and resi 11 and name CA)
color orange, FN_pair_6
hide labels, FN_pair_6
distance FN_pair_7, (chain A and resi 70 and name CA), (chain B and resi 35 and name CA)
color orange, FN_pair_7
hide labels, FN_pair_7
distance FN_pair_8, (chain A and resi 276 and name CA), (chain B and resi 35 and name CA)
color orange, FN_pair_8
hide labels, FN_pair_8
distance FN_pair_9, (chain A and resi 90 and name CA), (chain B and resi 9 and name CA)
color orange, FN_pair_9
hide labels, FN_pair_9
distance FN_pair_10, (chain A and resi 285 and name CA), (chain B and resi 27 and name CA)
color orange, FN_pair_10
hide labels, FN_pair_10

zoom TP_residues or FP_residues or FN_residues
orient
bg_color white
