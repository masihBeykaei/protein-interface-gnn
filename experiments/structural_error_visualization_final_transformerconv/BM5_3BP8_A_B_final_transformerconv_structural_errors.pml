reinitialize
load ../../data/raw_pdb_expanded_bm5/BM5_3BP8_A_B.pdb, BM5_3BP8_A_B
hide everything
show cartoon, BM5_3BP8_A_B
color gray80, BM5_3BP8_A_B
set cartoon_transparency, 0.25
set stick_radius, 0.22
set dash_width, 2.5
set dash_gap, 0.35
select TP_residues, (chain A and resi 557) or (chain A and resi 636) or (chain A and resi 637) or (chain A and resi 686) or (chain A and resi 687) or (chain A and resi 688) or (chain A and resi 906) or (chain B and resi 34) or (chain B and resi 60) or (chain B and resi 61) or (chain B and resi 62) or (chain B and resi 63) or (chain B and resi 64)
select FP_residues, (chain A and resi 633) or (chain A and resi 635) or (chain A and resi 637) or (chain A and resi 638) or (chain A and resi 639) or (chain A and resi 683) or (chain A and resi 685) or (chain A and resi 687) or (chain A and resi 906) or (chain B and resi 33) or (chain B and resi 59) or (chain B and resi 61) or (chain B and resi 64) or (chain B and resi 65) or (chain B and resi 73)
select FN_residues, (chain A and resi 584) or (chain A and resi 585) or (chain A and resi 640) or (chain A and resi 643) or (chain A and resi 644) or (chain A and resi 645) or (chain A and resi 646) or (chain A and resi 684) or (chain A and resi 906) or (chain B and resi 34) or (chain B and resi 35) or (chain B and resi 36) or (chain B and resi 40) or (chain B and resi 49) or (chain B and resi 68)
color green, TP_residues
color red, FP_residues
color orange, FN_residues
show sticks, TP_residues
show sticks, FP_residues
show sticks, FN_residues
distance TP_pair_1, (chain A and resi 637 and name CA), (chain B and resi 64 and name CA)
color green, TP_pair_1
hide labels, TP_pair_1
distance TP_pair_2, (chain A and resi 686 and name CA), (chain B and resi 61 and name CA)
color green, TP_pair_2
hide labels, TP_pair_2
distance TP_pair_3, (chain A and resi 686 and name CA), (chain B and resi 62 and name CA)
color green, TP_pair_3
hide labels, TP_pair_3
distance TP_pair_4, (chain A and resi 686 and name CA), (chain B and resi 63 and name CA)
color green, TP_pair_4
hide labels, TP_pair_4
distance TP_pair_5, (chain A and resi 687 and name CA), (chain B and resi 60 and name CA)
color green, TP_pair_5
hide labels, TP_pair_5
distance TP_pair_6, (chain A and resi 636 and name CA), (chain B and resi 64 and name CA)
color green, TP_pair_6
hide labels, TP_pair_6
distance TP_pair_7, (chain A and resi 688 and name CA), (chain B and resi 61 and name CA)
color green, TP_pair_7
hide labels, TP_pair_7
distance TP_pair_8, (chain A and resi 557 and name CA), (chain B and resi 34 and name CA)
color green, TP_pair_8
hide labels, TP_pair_8
distance TP_pair_9, (chain A and resi 687 and name CA), (chain B and resi 62 and name CA)
color green, TP_pair_9
hide labels, TP_pair_9
distance TP_pair_10, (chain A and resi 906 and name CA), (chain B and resi 34 and name CA)
color green, TP_pair_10
hide labels, TP_pair_10
distance FP_pair_1, (chain A and resi 637 and name CA), (chain B and resi 65 and name CA)
color red, FP_pair_1
hide labels, FP_pair_1
distance FP_pair_2, (chain A and resi 639 and name CA), (chain B and resi 64 and name CA)
color red, FP_pair_2
hide labels, FP_pair_2
distance FP_pair_3, (chain A and resi 635 and name CA), (chain B and resi 64 and name CA)
color red, FP_pair_3
hide labels, FP_pair_3
distance FP_pair_4, (chain A and resi 638 and name CA), (chain B and resi 64 and name CA)
color red, FP_pair_4
hide labels, FP_pair_4
distance FP_pair_5, (chain A and resi 687 and name CA), (chain B and resi 59 and name CA)
color red, FP_pair_5
hide labels, FP_pair_5
distance FP_pair_6, (chain A and resi 633 and name CA), (chain B and resi 64 and name CA)
color red, FP_pair_6
hide labels, FP_pair_6
distance FP_pair_7, (chain A and resi 906 and name CA), (chain B and resi 33 and name CA)
color red, FP_pair_7
hide labels, FP_pair_7
distance FP_pair_8, (chain A and resi 685 and name CA), (chain B and resi 61 and name CA)
color red, FP_pair_8
hide labels, FP_pair_8
distance FP_pair_9, (chain A and resi 906 and name CA), (chain B and resi 73 and name CA)
color red, FP_pair_9
hide labels, FP_pair_9
distance FP_pair_10, (chain A and resi 683 and name CA), (chain B and resi 61 and name CA)
color red, FP_pair_10
hide labels, FP_pair_10
distance FN_pair_1, (chain A and resi 643 and name CA), (chain B and resi 68 and name CA)
color orange, FN_pair_1
hide labels, FN_pair_1
distance FN_pair_2, (chain A and resi 906 and name CA), (chain B and resi 35 and name CA)
color orange, FN_pair_2
hide labels, FN_pair_2
distance FN_pair_3, (chain A and resi 584 and name CA), (chain B and resi 36 and name CA)
color orange, FN_pair_3
hide labels, FN_pair_3
distance FN_pair_4, (chain A and resi 684 and name CA), (chain B and resi 49 and name CA)
color orange, FN_pair_4
hide labels, FN_pair_4
distance FN_pair_5, (chain A and resi 585 and name CA), (chain B and resi 36 and name CA)
color orange, FN_pair_5
hide labels, FN_pair_5
distance FN_pair_6, (chain A and resi 644 and name CA), (chain B and resi 36 and name CA)
color orange, FN_pair_6
hide labels, FN_pair_6
distance FN_pair_7, (chain A and resi 645 and name CA), (chain B and resi 34 and name CA)
color orange, FN_pair_7
hide labels, FN_pair_7
distance FN_pair_8, (chain A and resi 644 and name CA), (chain B and resi 68 and name CA)
color orange, FN_pair_8
hide labels, FN_pair_8
distance FN_pair_9, (chain A and resi 640 and name CA), (chain B and resi 40 and name CA)
color orange, FN_pair_9
hide labels, FN_pair_9
distance FN_pair_10, (chain A and resi 646 and name CA), (chain B and resi 36 and name CA)
color orange, FN_pair_10
hide labels, FN_pair_10
zoom TP_residues or FP_residues or FN_residues
orient
bg_color white
# Optional high-resolution export:
# ray 1800, 1400
# png BM5_3BP8_A_B_final_transformerconv_structural_errors.png, dpi=300
