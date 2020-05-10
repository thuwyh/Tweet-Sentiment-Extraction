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

### todo
- [x] 星号相关的探索
- 多区间训练及数据增强
