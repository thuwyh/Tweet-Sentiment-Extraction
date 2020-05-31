cd ../src
# python train_v4.py validate test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--clean

# python train_v4.py train test_roberta4 --batch-size 16 --train-file train_roberta2.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post --smooth #--clean #--weight_decay 0.01 #
# python train_v4.py train test_roberta4 --batch-size 16 --train-file train_roberta2.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v4.py train test_roberta4 --batch-size 16 --train-file train_roberta2.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v4.py train test_roberta4 --batch-size 16 --train-file train_roberta2.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v4.py train test_roberta4 --batch-size 16 --train-file train_roberta2.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01

# python train_v4.py train test_roberta5 --batch-size 16 --train-file train_roberta2.pkl --fold 0 --bert-path ../../bert_models/finetuned_roberta/ --post #--clean #--weight_decay 0.01 #
# python train_v4.py train test_roberta5 --batch-size 16 --train-file train_roberta2.pkl --fold 1 --bert-path ../../bert_models/finetuned_roberta/ --post #--weight_decay 0.01
# python train_v4.py train test_roberta5 --batch-size 16 --train-file train_roberta2.pkl --fold 2 --bert-path ../../bert_models/finetuned_roberta/ --post #--weight_decay 0.01
# python train_v4.py train test_roberta5 --batch-size 16 --train-file train_roberta2.pkl --fold 3 --bert-path ../../bert_models/finetuned_roberta/ --post #--weight_decay 0.01
# python train_v4.py train test_roberta5 --batch-size 16 --train-file train_roberta2.pkl --fold 4 --bert-path ../../bert_models/finetuned_roberta/ --post #--weight_decay 0.01

# aug data
# python train_v5.py train v5 --batch-size 16 --fold 0 --bert-path ../../bert_models/roberta_base/ --post --smooth #--smooth #--clean #--weight_decay 0.01 #
# python train_v5.py train v5 --batch-size 16 --fold 1 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v5.py train v5 --batch-size 16 --fold 2 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v5.py train v5 --batch-size 16 --fold 3 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v5.py train v5 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01

# python train_v5.py validate v5 --batch-size 16 --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--clean #--weight_decay 0.01 #
# python train_v5.py validate v5 --batch-size 16 --fold 1 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v5.py validate v5 --batch-size 16 --fold 2 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v5.py validate v5 --batch-size 16 --fold 3 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v5.py validate v5 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01

# with old sentiment
# python train_v5.py predict5 v5 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v6.py train v6 --batch-size 16 --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--smooth #--clean #--weight_decay 0.01 #
# python train_v6.py train v6 --batch-size 16 --fold 1 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v6.py train v6 --batch-size 16 --fold 2 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v6.py train v6 --batch-size 16 --fold 3 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v6.py train v6 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01

# python train_v6.py validate v6 --batch-size 16 --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--smooth #--clean #--weight_decay 0.01 #
# python train_v6.py validate v6 --batch-size 16 --fold 1 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v6.py validate v6 --batch-size 16 --fold 2 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v6.py validate v6 --batch-size 16 --fold 3 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v6.py validate v6 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01

# python train_v7.py train v7 --batch-size 16 --fold 0 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--smooth #--clean #--weight_decay 0.01 #
# python train_v7.py train v7 --batch-size 16 --fold 1 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
# python train_v7.py train v7 --batch-size 16 --fold 2 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
# python train_v7.py train v7 --batch-size 16 --fold 3 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
# python train_v7.py train v7 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
# python train_v7.py validate5 v7 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01

# python train_v7.py validate v7 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01


# # distill
# python train_v7_distill.py train v7d --batch-size 16 --fold 0 --bert-path ../../bert_models/roberta_base/ --post --distill #--smooth #--smooth #--clean #--weight_decay 0.01 #
# python train_v7_distill.py train v7d --batch-size 16 --fold 1 --bert-path ../../bert_models/roberta_base/ --post --distill #--smooth #--weight_decay 0.01
# python train_v7_distill.py train v7d --batch-size 16 --fold 2 --bert-path ../../bert_models/roberta_base/ --post --distill #--smooth #--weight_decay 0.01
# python train_v7_distill.py train v7d --batch-size 16 --fold 3 --bert-path ../../bert_models/roberta_base/ --post --distill #--smooth #--weight_decay 0.01
# python train_v7_distill.py train v7d --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ --post --distill #--smooth #--weight_decay 0.01

python train_v8.py train v8 --batch-size 16 --fold 0 --bert-path ../../bert_models/roberta_base/  #--post #--smooth #--smooth #--clean #--weight_decay 0.01 #
python train_v8.py train v8 --batch-size 16 --fold 1 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
python train_v8.py train v8 --batch-size 16 --fold 2 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
python train_v8.py train v8 --batch-size 16 --fold 3 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
python train_v8.py train v8 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
# python train_v8.py validate5 v8 --batch-size 16 --fold 4 --bert-path ../../bert_models/roberta_base/ #--post #--smooth #--weight_decay 0.01
