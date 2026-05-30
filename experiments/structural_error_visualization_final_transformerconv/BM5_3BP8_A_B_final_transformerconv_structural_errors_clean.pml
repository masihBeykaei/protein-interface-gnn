reinitialize
load ../../data/raw_pdb_expanded_bm5/BM5_3BP8_A_B.pdb, BM5_3BP8_A_B
hide everything
show cartoon, BM5_3BP8_A_B
color gray80, BM5_3BP8_A_B
set cartoon_transparency, 0.30
set stick_radius, 0.28
select TP_residues, (chain A and resi 557) or (chain A and resi 636) or (chain A and resi 637) or (chain A and resi 686) or (chain A and resi 687) or (chain A and resi 688) or (chain A and resi 906) or (chain B and resi 34) or (chain B and resi 60) or (chain B and resi 61) or (chain B and resi 62) or (chain B and resi 63) or (chain B and resi 64)
select FP_residues, (chain A and resi 633) or (chain A and resi 635) or (chain A and resi 637) or (chain A and resi 638) or (chain A and resi 639) or (chain A and resi 683) or (chain A and resi 685) or (chain A and resi 687) or (chain A and resi 906) or (chain B and resi 33) or (chain B and resi 59) or (chain B and resi 61) or (chain B and resi 64) or (chain B and resi 65) or (chain B and resi 73)
select FN_residues, (chain A and resi 584) or (chain A and resi 585) or (chain A and resi 640) or (chain A and resi 643) or (chain A and resi 644) or (chain A and resi 645) or (chain A and resi 646) or (chain A and resi 684) or (chain A and resi 906) or (chain B and resi 34) or (chain B and resi 35) or (chain B and resi 36) or (chain B and resi 40) or (chain B and resi 49) or (chain B and resi 68)
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
# png BM5_3BP8_A_B_final_transformerconv_structural_errors_clean.png, dpi=300
