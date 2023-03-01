python run.py \
  --device 'cpu' \
  --oracle_model tape \
  --alg batchbo \
  --name 'cnn-batchbo-q100-ucb-tape-1-AAV' \
  --num_rounds 30 \
  --net cnn \
  --ensemble_size 3 \
  --out_dir result \



