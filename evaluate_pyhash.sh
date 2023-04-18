




for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_dnerf_pyhash_$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--dt_gamma 0 2>&1 | tee -a pyhash.log

done


for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_dnerf_pyhash_basis$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--dt_gamma 0 \
--basis \
2>&1 | tee -a pyhash.log

done


