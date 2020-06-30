cd ../src

for i in {0..9}
do
python train_v11.py train v11_10 --batch-size 16 --fold $i --bert-path ../../bert_models/roberta_base/ --fgm --post --train-file train_10fold.csv --offset 7
done
python train_v11.py validate5 v11_10 --batch-size 32 --fold 1 --bert-path ../../bert_models/roberta_base/ --post --train-file train_10fold.csv --offset 7
python train_v11.py predict5 v11_10 --batch-size 32 --fold 0 --bert-path ../../bert_models/roberta_base/ --post --train-file train_10fold.csv --offset 7
