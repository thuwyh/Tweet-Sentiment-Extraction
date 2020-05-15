### 2020.04.24
- 划分本地测试集用于测试融合情况
- 清理工程
- predict 输出概率，get_predicts输出预测的句子，evaluate进行评估
- 在预测结果上进行融合效果比在token层面进行融合效果好。0.7166 vs 0.7159
- 融合roberta和bert base，0.7171

### 2020.04.26
- 多模型直接在word logit上融合效果也不错
- 增加pseudo labeling相关代码

### 2020.04.28
- 修改预处理代码

### 2020.05.06
- fp16不好
- weight decay 不好
- roberta large不好
- 替换双引号不好
- 原始loss: 5 fold avg 0.70946
- 2倍loss：5 fold avg 0.70972

### 2020.05.07
- 星号似乎没有什么后处理的可能性
- 加入了多个可能区间的训练
- 两重improve+cnn：0.71034
- 两重improve，没有cnn：0.70946
- 一重improve+cnn：0.71074，完整5折：0.71135
- 没有space的improve+cnn：0.70964

### 2020.05.10
- mask掉sentiment增强没用
- 堆叠两个bert也没用
- 两层卷积也没用
- freeze embedding，完整5折,3 epoch: 0.71256; 4epoch变差明显
- freeze embedding+3 layers: failed
- cosine schedule: failed
- cosine with restart: 0.71235
- fgm: 0.713528
- freeze embedding+fgm: worse than above

### 2020.05.12(office)
- 只用训练集finetune lm, 0.71211
- 增加测试集finetune lm，0.71296
- label smoothing 暂时失败

### 2020.05.13(office)
- 修改前处理和后处理, 0.7149
- 修改前处理和后处理+finetune LM，
- (home)修改后数据+前后处理，0.7157

### 2020.05.14
- 清理数据（错误的end），0.71600

### todo
- [x] 星号相关的探索
- 多区间训练及数据增强


### finetune
python lm_finetune.py --train_data_file ../input/corpus.txt \
--output_dir ../../bert_models/finetuned_roberta/ \
--model_type roberta \
--line_by_line \
--model_name_or_path ../../bert_models/roberta_base/ \
--mlm \
--do_train \
--per_gpu_train_batch_size 16 \
--num_train_epochs 2 \
--save_total_limit 1

python lm_finetune.py --train_data_file ../input/corpus_aug.txt \
--output_dir ../../bert_models/finetuned_roberta_aug/ \
--model_type roberta \
--line_by_line \
--model_name_or_path ../../bert_models/roberta_base/ \
--mlm \
--do_train \
--per_gpu_train_batch_size 16 \
--num_train_epochs 2 \
--save_total_limit 1