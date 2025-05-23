export CUDA_VISIBLE_DEVICES=0
n=5000
for (( i=0; i<n; i+=1 )); do
  echo "python main.py --config config/base.yaml --experiment experiment_5x1 --signature $i --target ../../../MDS-HR/test/$i.png --log_dir test_dataset_log/"
  python main.py --config config/base.yaml --experiment experiment_5x1 --signature $i --target ../../../MDS-HR/test/$i.png --log_dir test_dataset_log/
done