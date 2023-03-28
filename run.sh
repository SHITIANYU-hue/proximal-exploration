python run.py \
  --device 'cpu' \
  --oracle_model antibody \
  --alg batchbo \
  --name 'cnn-batchbo-UCBrisk-q100-antibody_seed1-anti-p80' \
  --num_rounds 10 \
  --task E4B \
  --net cnn \
  --ensemble_size 3 \
  --out_dir resultantibody \



# python run.py \
#   --device 'cpu' \
#   --oracle_model antibody \
#   --alg pex \
#   --name 'cnn-pex-q100-antibody_seed1' \
#   --num_rounds 10 \
#   --task E4B \
#   --net cnn \
#   --ensemble_size 3 \
#   --out_dir resultantibody \


