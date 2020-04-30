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