cd ../src

for i in {0..9}
do
python train_v10.py train v10_10 --batch-size 16 --fold $i --bert-path ../../bert_models/roberta_base/ --fgm --post --train-file train_10fold.csv
done
python train_v10.py validate5 v10_10 --batch-size 32 --bert-path ../../bert_models/roberta_base/ --post --train-file train_10fold.csv
python train_v10.py predict5 v10_10 --batch-size 32 --bert-path ../../bert_models/roberta_base/ --post --train-file train_10fold.csv
