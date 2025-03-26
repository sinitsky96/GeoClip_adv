data=results/new
attack=sparse_rs_patches
name=sparse_rs_patches_geoclip_1_150_nqueries_32_pinit_0.30_loss_margin_eps_128_k_4_targeted_False_targetclass_None_seed_42

mamba activate cs236207
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv

# if geoclip, use --unnormalize
python ./sparse_rs/vis.py --adv_pth $data/$attack/$name.pth --out_pdf ./results/plots/$name.pdf