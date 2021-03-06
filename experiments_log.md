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

### 2020.05.15
- 4GM data+seed 42: 0.7148

### 2020.05.16
- 4GM data+seed 42+0.716sub: cv: 0.717372, lb: 0.714
- 4GM data+seed 42+finetune LM: cv 0.71509, lb: 0.715

### 2020.05.19
- 4GM data+seed 42+0.716sub smooth: cv 0.7165, lb: 0.716
- 4GM data+seed 42+0.716sub, all smooth, without fgm: cv 0.71588, lb 0.714
- 4GM data+seed 42+synonyms: cv 0.71562


### 2020.05.23
- v7: cv 0.7153, lb 0.718
- v7+distillation: cv 0.72386, lb 0.712，可能有leak

### 2020.05.24
- v7 处理特殊字符: cv 0.715688
- v7 distillation: cv 0.71714

### 2020.05.25
- v7 改网络结构: cv 0.71580

### 2020.05.31
- v8 增加原始数据集，简单修改了网络结构： cv 0.7161
- v8 不完全增加旧情感： cv 0.71622

### 2020.06.09
case study
- 对于shift=1的broken samples，有两种情况
    - 第一种是表现得像shift=2一样，例如
    ```
    It`s fun to see that glimpse of your life
    s fun
    ```
    - 第二种是前面多一个标点符号
    ```
    1  Time for me to seek out some coffee for my own caffein love affair too!! Mmmmm... Sweet been of hyper-goodness!!
    -goodness!
    ```
- 对于shift=2的broken samples， 也有两种情况
    - 第一种是多1个字符
    - 第二种是多两个字符
    
前处理规则
- shift=1: start pos 回退一位

后处理规则
- 对于shift=1的样本，只有第二种可以修
- 对于shift==2的样本，start pos统一滑动2个位置
- 对于shift>2的样本，start pos统一滑动shift-1个位置，end pos滑动shift-2个位置

尝试训练策略
- 如果是整句样本，不训练头、尾分类器，不起效。

修正了后处理的小bug：0.723594


### 2020.06.10
- 修正模型bug，cv 0.724118
- 添加空格数据， cv 0.72548

### 2020.06.12
- 10 fold: cv 0.725153, lb 0.720

### 2020.06.14
- 10 fold with old sentiment: cv 0.726468