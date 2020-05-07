cd ../src

python train_multi.py train roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--weight_decay 0.01 #--clean
python train_multi.py train roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post #--weight_decay 0.01
python train_multi.py train roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post #--weight_decay 0.01
python train_multi.py train roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post #--weight_decay 0.01
python train_multi.py train roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post #--weight_decay 0.01

# python train_multi.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta2_with_pseudo.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01 # --clean
# python train_multi.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta2_with_pseudo.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01
# python train_multi.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta2_with_pseudo.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01
# python train_multi.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta2_with_pseudo.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01
# python train_multi.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta2_with_pseudo.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01


# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--clean
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post 
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post 
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post 
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

# python train_multi.py validate5 roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post
# python train_multi.py validate52、 roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

# python train_multi.py predict5 roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --bert-path ../../bert_models/roberta_base/ --post

#### end #####


### bert base
# python train_multi.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 0 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3 #--clean
# python train_multi.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3 
# python train_multi.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3
# python train_multi.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3
# python train_multi.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3

# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--clean
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post 
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post 
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post 
# python train_multi.py validate roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

# python ensemble.py validate5 roberta_multi --batch-size 16 --train-file train_roberta2_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post
