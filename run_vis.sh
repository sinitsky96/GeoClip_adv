data=results
attack=sparse_rs_L0
name=sparse_rs_L0_clip_1_200_nqueries_100_pinit_0.30_loss_margin_eps_400_targeted_False_targetclass_None_seed_42.pth

pdf_outputname=sparse_rs_pixel_clip_eps400_nqueries_100

mamba activate cs236207
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv

# if geoclip, use --unnormalize
python ./sparse_rs/vis.py --adv_pth $data/$attack/$name --out_pdf ./results/plots/$pdf_outputname.pdf