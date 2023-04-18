




for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_dnerf_cuhash_$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--dt_gamma 0 2>&1 | tee -a cuhash.log

done


