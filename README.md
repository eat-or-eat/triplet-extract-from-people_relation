# triplet-extract-from-people_relation

一个简单的三元组数据的抽取任务，数据来源于其他github大神整理的人物关系数据

TODO

---

- [X]  基本baseline -21/12/22
- [ ]  模型调参优化，生成训练过程以及结果展示文件
- [ ]  上线到自己的云服务器测试

## 一，使用项目

环境：

```bash
matplotlib==3.3.4
numpy==1.20.1
pandas==1.2.4
pytorch_crf==0.7.2
torch==1.8.2+cu111
```

## 1.下载

`git clone`

## 2.（可选）处理自己的数据成本项目数据格输入式

```bash
处理成如下的xlsx文件，或者根据自己的三元组数据集修改src.loader中的读取部分代码

```




| 人物1  | 人物2  | 关系 | 文本                                                           |
| -------- | -------- | ------ | ---------------------------------------------------------------- |
| 韩庚   | 卢靖姗 | 夫妻 | 昨天有个热搜，韩庚和卢靖姗结婚了，将在31日举行婚礼。           |
| 巩俐   | 黄和祥 | 夫妻 | 婚后巩俐与黄和祥把家安在香港，过着普通人的平凡生活。           |
| 程砚秋 | 果素瑛 | 夫妻 | 程砚秋与果素瑛生有三子一女，即程永光、程永源、程永江和程慧贞。 |
| 程砚秋 | 程永光 | 父母 | 程砚秋与果素瑛生有三子一女，即程永光、程永源、程永江和程慧贞。 |

## 3.训练

`python main.py`

打印内容实例

```bash
2021-12-22 23:11:45,617 - __main__ - INFO - 测试第35轮模型效果:
2021-12-22 23:11:46,453 - __main__ - INFO - o1_acc : 0.2466494845360825 
2021-12-22 23:11:46,453 - __main__ - INFO - rel_acc : 0.2752577319587629 
2021-12-22 23:11:46,453 - __main__ - INFO - o2_acc : 0.17190721649484536 
2021-12-22 23:11:46,453 - __main__ - INFO - full_match_acc : 0.048195876288659796 
2021-12-22 23:11:46,453 - __main__ - INFO - --------------------
```

## 4.预测

`python predict.py`

打印内容示例

```bash
模型加载完毕!
('徐天明', 'unknown', '金民哲')
('张三', 'unknown', '李四')

```

# 二，项目介绍

```bash
 │  config.py 
│  main.py
│  predict.py  # 预测部分代码
│  README.md
│  requirements.txt
│
├─data
│      rel_dict.json  # 关系的scheme
│      vocab.txt  # bert-chese的词表，后面有空试下bert，效果应该会提升很多
│      人物关系表.xlsx  # 最下面数据集链接获取的数据
│
├─output
│  └─model
│          epoch_15.pth
│          epoch_35.pth
│
├─src
│  │  evaluator.py  # 测试脚本
│  │  loader.py  # 加载数据脚本
│  │  model.py  # 模型结构
```

数据来源:[这个项目中的人物关系表.xlsx和rel_dict.json](https://github.com/percent4/people_relation_extract/tree/master/data)
