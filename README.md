# 文本多分类、序列标注任务 
基于BERT的中文情感分类任务
基于BERT、LSTM、CRF 的中文序列标注任务

#### 文本分类
bert-dense
```
这里的文本分类主要是多分类，如果是二分类任务可以自己替换损失函数
``` 


#### 序列标注
bert-bilstm-crf 序列标注任务
```
pytorch 微调 bert 模型 应用于下游分类、序列标注任务，
bert模块使用的是hugging face 发布的第三方库[transformers](https://huggingface.co/transformers/)   
crf模块参考了[pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable/)的内容，做了部分修改，方便计算准确率
```

##### 环境配置
```
pip install -r requirements.txt
```

##### 训练参数配置
```
config.py  

序列标注需要修改自己的标签  SequenceLabelConfig.TAG_TO_ID
```


##### 数据准备
```
参考 data/README.md 文件 

```

##### 训练模型
```
文本分类  python text-classifizer.py
序列标注  python sequence-label.py
```

##### 查看训练过程日志记录(tensorboard)
```
tensorboard.exe --logdir=logs
```


##### 模型训练，验证结果：
|        | training_acc  |  training_loss |   eval_acc | eval_loss | 
|  ----  | ----  | ----|  ----|  ----| 
| 文本分类  | 0.9766 |0.07909 |0.9922 |  0.0868|
| 序列标注  | 0.9838 |19.706 | 0.9175|  38.77| 


