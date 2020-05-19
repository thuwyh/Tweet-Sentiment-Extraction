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

# pseudo+smooth pseudo
# python train_v4.py train 4gm_p_s --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post --smooth #--clean #--weight_decay 0.01 #
# python train_v4.py train 4gm_p_s --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v4.py train 4gm_p_s --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v4.py train 4gm_p_s --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
# python train_v4.py train 4gm_p_s --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01

# pseudo + smooth all no fgm
python train_v4.py train 4gm_p_s_nofgm --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post --smooth --clean #--weight_decay 0.01 #
python train_v4.py train 4gm_p_s_nofgm --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
python train_v4.py train 4gm_p_s_nofgm --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
python train_v4.py train 4gm_p_s_nofgm --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01
python train_v4.py train 4gm_p_s_nofgm --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post --smooth #--weight_decay 0.01

# pseudo
# python train_v4.py train test_roberta6_pseudo --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--clean #--weight_decay 0.01 #
# python train_v4.py train test_roberta6_pseudo --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v4.py train test_roberta6_pseudo --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v4.py train test_roberta6_pseudo --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01
# python train_v4.py train test_roberta6_pseudo --batch-size 16 --train-file train_roberta4_pseudo.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--weight_decay 0.01

# python train_v4.py predict test_roberta6 --batch-size 16 --test-file train.csv --train-file train_roberta3.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--smooth #--clean #--weight_decay 0.01 #

# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 0 --bert-path ../../bert_models/finetuned_roberta/ --post --clean #--no-neutral #--clean #--weight_decay 0.01 #
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 1 --bert-path ../../bert_models/finetuned_roberta/ --post #--no-neutral #--weight_decay 0.01
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 2 --bert-path ../../bert_models/finetuned_roberta/ --post #--no-neutral #--weight_decay 0.01
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 3 --bert-path ../../bert_models/finetuned_roberta/ --post #--no-neutral #--weight_decay 0.01
# python train_v3.py train test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 4 --bert-path ../../bert_models/finetuned_roberta/ --post #--no-neutral #--weight_decay 0.01

# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01 # --clean
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01
# python train_v3.py train test3_roberta_pseudo --batch-size 16 --train-file train_roberta_with_pseudo.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post --weight_decay 0.01


# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--clean
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta2.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

# python train_v3.py validate5 test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post
# python train_v3.py validate52、 test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

# python train_v3.py predict5 test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --bert-path ../../bert_models/roberta_base/ --post

#### end #####


### bert base
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 0 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3 #--clean
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 1 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3 
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 2 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 3 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3
# python train_v3.py train test3_bert --batch-size 16 --train-file train_bert.pkl --fold 4 --bert-path ../../bert_models/bert_base_uncased/ --post --weight_decay 0.01 --offset 3

# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 0 --bert-path ../../bert_models/roberta_base/ --post #--clean
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 1 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 2 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 3 --bert-path ../../bert_models/roberta_base/ --post 
# python train_v3.py validate test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post

# python ensemble.py validate5 test3_roberta3 --batch-size 16 --train-file train_roberta_v5.pkl --fold 4 --bert-path ../../bert_models/roberta_base/ --post
