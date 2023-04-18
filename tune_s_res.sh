



for reso in 32 64 128 256 512 1024
do

for size in 19 20 21
do
sce="standup"

python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_dnerf_pyhash_$sce$reso\size$size \
-O \
--bound \
1.0  \
--scale \
0.8 \
--log2_hashmap_size $size \
--spatial_reso $reso \
--dt_gamma 0 2>&1 | tee -a tune.log

done




done
