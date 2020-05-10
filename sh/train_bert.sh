cd ../src

python train_v3.py train bert_base --batch-size 16 --train-file train_bert2.pkl --fold 0 --bert-path ../../bert_models/bert_base_uncased/ --post --offset 3 #--clean #--weight_decay 0.01 #
python train_v3.py train bert_base --batch-size 16 --train-file train_bert2.pkl --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --post --offset 3 #--weight_decay 0.01
python train_v3.py train bert_base --batch-size 16 --train-file train_bert2.pkl --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --post --offset 3 #--weight_decay 0.01
python train_v3.py train bert_base --batch-size 16 --train-file train_bert2.pkl --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --post --offset 3 #--weight_decay 0.01
python train_v3.py train bert_base --batch-size 16 --train-file train_bert2.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post --offset 3 #--weight_decay 0.01

# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 0 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 # --clean
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01


# python train_v3.py validate bert_base --batch-size 16 --train-file train_bert2.pkl --fold 0 --bert-path ../../bert_models/bert_base_uncased/ --post #--clean
# python train_v3.py validate bert_base --batch-size 16 --train-file train_bert2.pkl --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --post 
# python train_v3.py validate bert_base --batch-size 16 --train-file train_bert2.pkl --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --post 
# python train_v3.py validate bert_base --batch-size 16 --train-file train_bert2.pkl --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --post 
# python train_v3.py validate bert_base --batch-size 16 --train-file train_bert2.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post

# python train_v3.py validate5 bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post
# python train_v3.py validate52、 bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post

# python train_v3.py predict5 bert_base --batch-size 16 --train-file train_roberta_v5.pkl --bert-path ../../bert_models/bert_base_uncased/ --post

#### end #####


### bert base
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 0 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3 #--clean
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3 
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3

# python train_v3.py validate bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 0 --bert-path ../../bert_models/bert_base_uncased/ --post #--clean
# python train_v3.py validate bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --post 
# python train_v3.py validate bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --post 
# python train_v3.py validate bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --post 
# python train_v3.py validate bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post

# python ensemble.py validate5 bert_base --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post
