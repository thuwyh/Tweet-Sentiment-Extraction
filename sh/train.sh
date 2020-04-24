cd ../src
# python train.py train test --batch-size 16 --clean --lr 5e-5
# python train.py validate test --batch-size 16

# python train3.py train test3 --batch-size 16 --bert-path ../../bert_models/bert_base_uncased/ --train-file train_v2.pkl
# python train3.py validate test3 --batch-size 16 --bert-path ../../bert_models/bert_base_uncased/
# python train3.py train test3 --batch-size 16 --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --train-file train_v2.pkl
# python train3.py train test3 --batch-size 16 --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --train-file train_v2.pkl
# python train3.py train test3 --batch-size 16 --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --train-file train_v2.pkl
# python train3.py train test3 --batch-size 16 --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --train-file train_v2.pkl


# python train3.py validate test3 --batch-size 16 --post
# python train3_roberta.py train test3_roberta2 --batch-size 16 --clean #--weight_decay 0.01
# # python train3_roberta.py train test3_roberta2 --batch-size 16 --fold 1 #--lr 2e-5 #--weight_decay 0.01
# python train3_roberta.py train test3_roberta2 --batch-size 16 --fold 2 #--weight_decay 0.01
# python train3_roberta.py train test3_roberta2 --batch-size 16 --fold 3 #--weight_decay 0.01
# python train3_roberta.py train test3_roberta2 --batch-size 16 --fold 4
# python train3_roberta.py validate test3_roberta2 --batch-size 16 --fold 4 --post

# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --bert-path ../../bert_models/roberta_base/ --post #--clean
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post 
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post 
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post 
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post 


##### in selected_text #####
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01 --fp16 #--clean
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post  --weight_decay 0.01  --fp16
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post  --weight_decay 0.01  --fp16
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post  --weight_decay 0.01  --fp16
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01 --fp16

# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 0 --bert-path ../../bert_models/roberta_large/ --post --weight_decay 0.01 --fp16 #--clean
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 1 --bert-path ../../bert_models/roberta_large/ --post --weight_decay 0.01 --fp16 
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 2 --bert-path ../../bert_models/roberta_large/ --post --weight_decay 0.01 --fp16 
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 3 --bert-path ../../bert_models/roberta_large/ --post --weight_decay 0.01 --fp16 
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_large/ --post --weight_decay 0.01 --fp16

# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --bert-path ../../bert_models/roberta_base/ --post #--clean
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

# python train_v3.py validate5 test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

python train_v3.py predict5 test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --bert-path ../../bert_models/roberta_base/ --post

#### end #####

# python train3_roberta.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --bert-path ../../bert_models/roberta_base/ --post #--clean
# python train3_roberta.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --bert-path ../../bert_models/roberta_base/ --post --fold 1 #--clean
# python train3_roberta.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --bert-path ../../bert_models/roberta_base/ --post --fold 2 #--clean
# python train3_roberta.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --bert-path ../../bert_models/roberta_base/ --post --fold 3 #--clean
# python train3_roberta.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --bert-path ../../bert_models/roberta_base/ --post --fold 4 #--clean

# python train3_roberta.py predict5 test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post 

# python train_roberta_v2.py train test4_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl #--bert-path ../../bert_models/roberta_base/
# python train_roberta_v2.py train test4_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --bert-path ../../bert_models/roberta_base/ --fold 1
# python train_roberta_v2.py train test4_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --bert-path ../../bert_models/roberta_base/ --fold 2
# python train_roberta_v2.py train test4_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --bert-path ../../bert_models/roberta_base/ --fold 3
# python train_roberta_v2.py train test4_roberta3 --batch-size 16 --train-file train_roberta_v4.pkl --bert-path ../../bert_models/roberta_base/ --fold 4

# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --lr 2e-5 #--clean
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --fold 1 --lr 2e-5
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --fold 2 --lr 2e-5
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --fold 3 --lr 2e-5
# python train3_roberta.py train test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --fold 4 --lr 2e-5
# python train3_roberta.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v3.pkl --post
# python train3_roberta.py predict5 test3_roberta2 --batch-size 16 --limit 100

# python train3_roberta.py train test3_roberta --batch-size 16 --max_grad_norm 1.0 #--clean
# python train3_roberta.py train test3_roberta --batch-size 16 --fold 1 --max_grad_norm 1.0
# python train3_roberta.py train test3_roberta --batch-size 16 --fold 2 --max_grad_norm 1.0
# python train3_roberta.py train test3_roberta --batch-size 16 --fold 3 --max_grad_norm 1.0
# python train3_roberta.py train test3_roberta --batch-size 16 --fold 4 --max_grad_norm 1.0
# python train3_roberta.py predict5 test3_roberta --batch-size 16 --limit 100

# python train3.py predict test3 --batch-size 16 --limit 100

# python train3.py predict5 test3 --batch-size 16 --limit 100
# concat
# python cat_roberta.py train cat_roberta --batch-size 16 --clean