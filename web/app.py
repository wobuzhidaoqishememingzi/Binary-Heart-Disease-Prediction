import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 1) 让 Python 能 import 到项目根目录的 ecs171model3.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../web
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../ecs171_project
sys.path.insert(0, PROJECT_ROOT)

import ecs171model3

# 2) 关键补丁
import __main__
__main__.PreprocessArtifacts = ecs171model3.PreprocessArtifacts

# 3) 加载 model + preprocess
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs_model3")
model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
preprocess = pickle.load(open(os.path.join(MODEL_DIR, "preprocess.pkl"), "rb"))

# 4) feature importance
fi_df = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index_col=0)
feature_importance = {
    "labels": fi_df.index.tolist()[:10],
    "values": [round(v, 4) for v in fi_df["importance"].tolist()[:10]],
}

# 5) 模型指标
with open(os.path.join(MODEL_DIR, "metrics.txt"), "r") as f:
    metrics_raw = f.read()
model_metrics = {}
for line in metrics_raw.split("\n"):
    if "Accuracy" in line and ":" in line:
        model_metrics["accuracy"] = float(line.split(":")[-1].strip())
    elif "Precision" in line and ":" in line and "macro" not in line:
        model_metrics["precision"] = float(line.split(":")[-1].strip())
    elif "Recall" in line and ":" in line and "macro" not in line:
        model_metrics["recall"] = float(line.split(":")[-1].strip())
    elif "F1" in line and ":" in line and "macro" not in line:
        model_metrics["f1"] = float(line.split(":")[-1].strip())

# 6) 训练数据分布（用于 Data Exploration tab）
raw_csv_path = os.path.join(PROJECT_ROOT, "heart_statlog_cleveland_hungary_final.csv")
raw_df = pd.read_csv(raw_csv_path)

def compute_distributions():
    """计算训练数据的分布，用于前端图表"""
    dist = {}
    # 连续变量: 按 target 分组的直方图
    cont_cols = ["age", "cholesterol", "resting bp s", "max heart rate", "oldpeak"]
    for col in cont_cols:
        grp0 = raw_df[raw_df["target"] == 0][col].dropna().tolist()
        grp1 = raw_df[raw_df["target"] == 1][col].dropna().tolist()
        dist[col] = {"no_disease": grp0, "disease": grp1}
    # target 比例
    vc = raw_df["target"].value_counts()
    dist["target_counts"] = {"no_disease": int(vc.get(0, 0)), "disease": int(vc.get(1, 0))}
    # 性别分布
    sex_target = raw_df.groupby(["sex", "target"]).size().reset_index(name="count")
    dist["sex_target"] = sex_target.to_dict(orient="records")
    return dist

data_distributions = compute_distributions()

# 7) 计算训练集每个 processed feature 的均值（用于 per-patient explanation）
processed_csv = os.path.join(PROJECT_ROOT, "heart_disease_processed.csv")
processed_df = pd.read_csv(processed_csv)
X_train_all = processed_df.drop("target", axis=1)
feature_means = X_train_all.mean().to_dict()


def compute_feature_contributions(df_processed):
    """
    Perturbation-based per-patient explanation:
    对每个特征，替换成训练集均值，看概率变化多少
    """
    base_proba = model.predict_proba(df_processed)[0][1]
    contributions = []
    feature_names = df_processed.columns.tolist()

    for feat in feature_names:
        perturbed = df_processed.copy()
        perturbed[feat] = feature_means.get(feat, 0)
        new_proba = model.predict_proba(perturbed)[0][1]
        diff = base_proba - new_proba  # positive = this feature pushes toward disease
        contributions.append({
            "feature": feat,
            "contribution": round(float(diff) * 100, 2),  # 转成百分比
        })

    # 按绝对值排序，取 top 8
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:8]


def analyze_risk_factors(input_dict):
    """分析用户输入的哪些指标异常"""
    risks = []
    if input_dict["age"] > 55:
        risks.append({"factor": "Age", "value": f'{int(input_dict["age"])} years', "note": "Age > 55 increases risk"})
    if input_dict["resting bp s"] > 140:
        risks.append({"factor": "Blood Pressure", "value": f'{int(input_dict["resting bp s"])} mmHg', "note": "Hypertension (> 140 mmHg)"})
    if input_dict["cholesterol"] > 240:
        risks.append({"factor": "Cholesterol", "value": f'{int(input_dict["cholesterol"])} mg/dl', "note": "High cholesterol (> 240 mg/dl)"})
    if input_dict["max heart rate"] < 100:
        risks.append({"factor": "Max Heart Rate", "value": f'{int(input_dict["max heart rate"])} bpm', "note": "Low max heart rate (< 100 bpm)"})
    if input_dict["oldpeak"] > 2:
        risks.append({"factor": "Oldpeak", "value": f'{input_dict["oldpeak"]}', "note": "High ST depression (> 2.0)"})
    if input_dict["exercise angina"] == 1:
        risks.append({"factor": "Exercise Angina", "value": "Yes", "note": "Exercise induced chest pain"})
    if input_dict["chest pain type"] == 4:
        risks.append({"factor": "Chest Pain", "value": "Asymptomatic", "note": "Asymptomatic chest pain type"})
    return risks


@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_importance=json.dumps(feature_importance),
        model_metrics=json.dumps(model_metrics),
        data_distributions=json.dumps(data_distributions),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """AJAX 预测端点，返回 JSON"""
    data = request.get_json()
    input_dict = {
        "age": float(data["age"]),
        "sex": int(data["sex"]),
        "chest pain type": int(data["chest_pain"]),
        "resting bp s": float(data["rest_bp"]),
        "cholesterol": float(data["chol"]),
        "fasting blood sugar": int(data["fbs"]),
        "resting ecg": int(data["rest_ecg"]),
        "max heart rate": float(data["max_hr"]),
        "exercise angina": int(data["ex_ang"]),
        "oldpeak": float(data["oldpeak"]),
        "ST slope": int(data["st_slope"]),
    }

    df_input = pd.DataFrame([input_dict])
    df_processed = ecs171model3.preprocess_transform(df_input, preprocess, has_target=False)

    pred = int(model.predict(df_processed)[0])
    proba = model.predict_proba(df_processed)[0]
    risk_pct = round(float(proba[1]) * 100, 1)

    contributions = compute_feature_contributions(df_processed)
    risk_factors = analyze_risk_factors(input_dict)

    return jsonify({
        "result": pred,
        "risk_pct": risk_pct,
        "label": "Heart Disease Detected" if pred == 1 else "No Heart Disease",
        "contributions": contributions,
        "risk_factors": risk_factors,
        "input_summary": {
            "Age": int(input_dict["age"]),
            "Sex": "Male" if input_dict["sex"] == 1 else "Female",
            "BP": int(input_dict["resting bp s"]),
            "Chol": int(input_dict["cholesterol"]),
            "Max HR": int(input_dict["max heart rate"]),
            "Oldpeak": input_dict["oldpeak"],
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=8080)
