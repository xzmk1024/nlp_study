import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

# 加载 Hugging Face 公开数据集：clue/clue
raw_ds = load_dataset("clue/clue", "tnews")

train_ds = raw_ds["train"]
val_ds = raw_ds["validation"]
test_ds = raw_ds["test"]

print("Dataset loaded:")
print(raw_ds)

# 抽样减少训练数据量
# 你可以调整下面的数字：
# train: 3000 条
# val:   500 条
# test:  500 条
# train_ds = train_ds.shuffle(seed=42).select(range(3000))
# val_ds = val_ds.shuffle(seed=42).select(range(500))
# test_ds  = test_ds.shuffle(seed=42).select(range(500))



print("\nAfter sampling:")
print("train size:", len(train_ds))
print("val size:", len(val_ds))
print("test size:", len(test_ds))


MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

num_labels = train_ds.features["label"].num_classes
print("\nnum_labels =", num_labels)

# 加载 bert-base 模型，并把最后的分类层改成 num_labels 输出
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)


#  分词处理（tokenize）
# 把文本转成 BERT 能吃的 input_ids 和 attention_mask
# truncation=True：文本过长就截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大长度（越小越快）
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=32
    )

# batched=True 表示一次处理一批数据，加速
train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)


#  label 字段重命名：label -> labels
# Hugging Face Trainer 默认要求标签字段叫 "labels"
train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")


#  设置数据格式为 torch，并只保留需要的字段
# input_ids：token id 序列
# attention_mask：注意力掩码（pad 的部分是 0）
# labels：真实类别
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


#  评估指标函数：accuracy（准确率）
# eval_pred 是 Trainer 传入的：
# (logits, labels)
# logits 是模型输出的未归一化分数
# 取 argmax 就是预测类别
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=1,                  # 训练的总轮数
    per_device_train_batch_size=32,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=32,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)


# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()