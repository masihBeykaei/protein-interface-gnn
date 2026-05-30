# PyMOL structural error visualization for 1BRS_A_B
# Green = TP, Red = FP, Orange = FN

reinitialize
load ../../data/raw_pdb/1BRS.pdb, 1BRS_A_B

hide everything
show cartoon
color gray80
set cartoon_transparency, 0.25
set stick_radius, 0.22
set dash_width, 2.5
set dash_gap, 0.35

select TP_residues, (chain A and resi 107) or (chain A and resi 108) or (chain A and resi 109) or (chain A and resi 110) or (chain A and resi 98) or (chain B and resi 107) or (chain B and resi 108) or (chain B and resi 109) or (chain B and resi 110) or (chain B and resi 98)
select FP_residues, (chain A and resi 107) or (chain A and resi 108) or (chain A and resi 97) or (chain A and resi 98) or (chain B and resi 106) or (chain B and resi 107) or (chain B and resi 108) or (chain B and resi 97)
select FN_residues, (chain A and resi 108) or (chain A and resi 12) or (chain A and resi 15) or (chain A and resi 8) or (chain B and resi 108) or (chain B and resi 110) or (chain B and resi 12) or (chain B and resi 8)

color green, TP_residues
color red, FP_residues
color orange, FN_residues

show sticks, TP_residues
show sticks, FP_residues
show sticks, FN_residues

distance TP_pair_1, (chain A and resi 109 and name CA), (chain B and resi 108 and name CA)
color green, TP_pair_1
hide labels, TP_pair_1
distance TP_pair_2, (chain A and resi 107 and name CA), (chain B and resi 98 and name CA)
color green, TP_pair_2
hide labels, TP_pair_2
distance TP_pair_3, (chain A and resi 108 and name CA), (chain B and resi 110 and name CA)
color green, TP_pair_3
hide labels, TP_pair_3
distance TP_pair_4, (chain A and resi 98 and name CA), (chain B and resi 107 and name CA)
color green, TP_pair_4
hide labels, TP_pair_4
distance TP_pair_5, (chain A and resi 109 and name CA), (chain B and resi 110 and name CA)
color green, TP_pair_5
hide labels, TP_pair_5
distance TP_pair_6, (chain A and resi 109 and name CA), (chain B and resi 107 and name CA)
color green, TP_pair_6
hide labels, TP_pair_6
distance TP_pair_7, (chain A and resi 108 and name CA), (chain B and resi 109 and name CA)
color green, TP_pair_7
hide labels, TP_pair_7
distance TP_pair_8, (chain A and resi 109 and name CA), (chain B and resi 109 and name CA)
color green, TP_pair_8
hide labels, TP_pair_8
distance TP_pair_9, (chain A and resi 110 and name CA), (chain B and resi 108 and name CA)
color green, TP_pair_9
hide labels, TP_pair_9
distance TP_pair_10, (chain A and resi 110 and name CA), (chain B and resi 110 and name CA)
color green, TP_pair_10
hide labels, TP_pair_10
distance FP_pair_1, (chain A and resi 107 and name CA), (chain B and resi 108 and name CA)
color red, FP_pair_1
hide labels, FP_pair_1
distance FP_pair_2, (chain A and resi 107 and name CA), (chain B and resi 97 and name CA)
color red, FP_pair_2
hide labels, FP_pair_2
distance FP_pair_3, (chain A and resi 108 and name CA), (chain B and resi 108 and name CA)
color red, FP_pair_3
hide labels, FP_pair_3
distance FP_pair_4, (chain A and resi 107 and name CA), (chain B and resi 107 and name CA)
color red, FP_pair_4
hide labels, FP_pair_4
distance FP_pair_5, (chain A and resi 108 and name CA), (chain B and resi 97 and name CA)
color red, FP_pair_5
hide labels, FP_pair_5
distance FP_pair_6, (chain A and resi 98 and name CA), (chain B and resi 108 and name CA)
color red, FP_pair_6
hide labels, FP_pair_6
distance FP_pair_7, (chain A and resi 97 and name CA), (chain B and resi 108 and name CA)
color red, FP_pair_7
hide labels, FP_pair_7
distance FP_pair_8, (chain A and resi 98 and name CA), (chain B and resi 97 and name CA)
color red, FP_pair_8
hide labels, FP_pair_8
distance FP_pair_9, (chain A and resi 107 and name CA), (chain B and resi 106 and name CA)
color red, FP_pair_9
hide labels, FP_pair_9
distance FP_pair_10, (chain A and resi 97 and name CA), (chain B and resi 97 and name CA)
color red, FP_pair_10
hide labels, FP_pair_10
distance FN_pair_1, (chain A and resi 15 and name CA), (chain B and resi 110 and name CA)
color orange, FN_pair_1
hide labels, FN_pair_1
distance FN_pair_2, (chain A and resi 108 and name CA), (chain B and resi 8 and name CA)
color orange, FN_pair_2
hide labels, FN_pair_2
distance FN_pair_3, (chain A and resi 108 and name CA), (chain B and resi 12 and name CA)
color orange, FN_pair_3
hide labels, FN_pair_3
distance FN_pair_4, (chain A and resi 12 and name CA), (chain B and resi 108 and name CA)
color orange, FN_pair_4
hide labels, FN_pair_4
distance FN_pair_5, (chain A and resi 8 and name CA), (chain B and resi 108 and name CA)
color orange, FN_pair_5
hide labels, FN_pair_5

zoom TP_residues or FP_residues or FN_residues
orient
bg_color white
