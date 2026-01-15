import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from openai import OpenAI
from fastapi import FastAPI, Query
import uvicorn

from typing import Dict
import threading
import time
import webbrowser

app = FastAPI()

# ---------------- 数据加载 ----------------
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

# 中文分词
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(str(x))))

# CountVectorizer 特征（适合 KNN）
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

# TF-IDF 特征（适合 NB、LR、SVM）
tfidf_vector = TfidfVectorizer()
tfidf_vector.fit(input_sentence.values)
input_feature_tfidf = tfidf_vector.transform(input_sentence.values)

# 1. KNN
knn_model = KNeighborsClassifier()
knn_model.fit(input_feature, dataset[1].values)

# 2. 朴素贝叶斯
nb_model = MultinomialNB()
nb_model.fit(input_feature_tfidf, dataset[1].values)

# 3. 逻辑回归
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(input_feature_tfidf, dataset[1].values)

# 4. 支持向量机
svm_model = LinearSVC()
svm_model.fit(input_feature_tfidf, dataset[1].values)

# ---------------- LLM 客户端 ----------------
client = OpenAI(
    api_key="sk-95d76819e3694838834b8a1bb09350a4",  # token
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ---------------- 接口定义 ----------------
@app.get("/text-cls/ml")
def text_classify_using_ml(
    model: str = Query(None, description="选择模型: knn / nb / lr / svm，可为空"),
    text: str = Query(..., description="待分类文本")
) -> Dict:
    test_sentence = " ".join(jieba.lcut(text))

    results = {}

    if model is None:
        # 同时运行所有模型
        results["knn"] = knn_model.predict(vector.transform([test_sentence]))[0]
        results["nb"] = nb_model.predict(tfidf_vector.transform([test_sentence]))[0]
        results["lr"] = lr_model.predict(tfidf_vector.transform([test_sentence]))[0]
        results["svm"] = svm_model.predict(tfidf_vector.transform([test_sentence]))[0]
        return {"method": "机器学习(全部)", "input": text, "predictions": results}

    elif model == "knn":
        prediction = knn_model.predict(vector.transform([test_sentence]))[0]
        return {"method": "KNN", "input": text, "prediction": prediction}

    elif model == "nb":
        prediction = nb_model.predict(tfidf_vector.transform([test_sentence]))[0]
        return {"method": "朴素贝叶斯", "input": text, "prediction": prediction}

    elif model == "lr":
        prediction = lr_model.predict(tfidf_vector.transform([test_sentence]))[0]
        return {"method": "逻辑回归", "input": text, "prediction": prediction}

    elif model == "svm":
        prediction = svm_model.predict(tfidf_vector.transform([test_sentence]))[0]
        return {"method": "SVM", "input": text, "prediction": prediction}

    else:
        return {"error": "模型类型错误，请选择 knn / nb / lr / svm"}


@app.get("/text-cls/llm")
def text_classify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型方法）
    """
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

                   输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
                   FilmTele-Play            
                   Video-Play               
                   Music-Play              
                   Radio-Listen           
                   Alarm-Update        
                   Travel-Query        
                   HomeAppliance-Control  
                   Weather-Query          
                   Calendar-Query      
                   TVProgram-Play      
                   Audio-Play       
                   Other             
                   """},  # 用户的提问
        ]
    )
    prediction = completion.choices[0].message.content
    return f"大模型预测类别: {prediction}"

def open_browser():
    time.sleep(8)
    file_path = os.path.abspath("test.html")
    webbrowser.open(f"file://{file_path}")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    uvicorn.run(
        "classify:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

