# Batch renderer for final tuned TransformerConv structural visualizations
# Run this file while PyMOL's current directory is:
# D:/Programing/BIOLOGY/protein_interface_gnn/experiments/structural_error_visualization_final_transformerconv
#
# Execute in PyMOL with:
# @render_all_final_transformerconv_clean.pml

reinitialize
@1BRS_A_B_final_transformerconv_structural_errors_clean.pml
bg_color white
set cartoon_transparency, 0.30
set stick_radius, 0.28
set ray_opaque_background, off
set antialias, 2
ray 1800, 1400
png 1BRS_A_B_final_transformerconv_clean.png, dpi=300

reinitialize
@1FSS_A_B_final_transformerconv_structural_errors_clean.pml
bg_color white
set cartoon_transparency, 0.30
set stick_radius, 0.28
set ray_opaque_background, off
set antialias, 2
ray 1800, 1400
png 1FSS_A_B_final_transformerconv_clean.png, dpi=300

reinitialize
@3HMX_LH_AB_final_transformerconv_structural_errors_clean.pml
bg_color white
set cartoon_transparency, 0.30
set stick_radius, 0.28
set ray_opaque_background, off
set antialias, 2
ray 1800, 1400
png 3HMX_LH_AB_final_transformerconv_clean.png, dpi=300

reinitialize
@BM5_1A2K_A_B_final_transformerconv_structural_errors_clean.pml
bg_color white
set cartoon_transparency, 0.30
set stick_radius, 0.28
set ray_opaque_background, off
set antialias, 2
ray 1800, 1400
png BM5_1A2K_A_B_final_transformerconv_clean.png, dpi=300

reinitialize
@BM5_3BP8_A_B_final_transformerconv_structural_errors_clean.pml
bg_color white
set cartoon_transparency, 0.30
set stick_radius, 0.28
set ray_opaque_background, off
set antialias, 2
ray 1800, 1400
png BM5_3BP8_A_B_final_transformerconv_clean.png, dpi=300

reinitialize
print "All five final TransformerConv clean renders have been exported."
