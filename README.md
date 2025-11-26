# SaveUs — 비만 위험도 모델(Obesity Risk Score)

본 문서는 SaveUs 서비스에서 사용되는 **새로운 비만 위험도(Obesity Risk Score) 모델**입니다.  
이번 버전은 “오늘 먹은 식단 기반 위험도 0~100점 계산”을 핵심 목표로 설계되었습니다.

---

## 1. 모델 목표

- 식단을 많이 먹으면 위험도가 올라가는 **직관적인 구조**  
- 결과는 **0~100 사이 연속값**  
- 의료·역학적 기준을 반영한 **명확한 가중치 기반 점수 공식**  
- 식단 데이터가 즉시 반영되는 **실시간 분석 모델**  

---

## 2. 위험도 점수 산출 공식 (Risk Score Formula)

비만 위험도는 다음 네 가지 영양 성분을 기준으로 계산합니다:

```text
risk_score =
    (total_calories / 2500 * 30) +
    (total_fat / 70 * 25) +
    (total_sugar / 50 * 20) +
    (total_sodium / 2000 * 25)
```

---

## 3. 공식 설명 및 가중치 근거

### 각 항목 설명

| 항목           | 기준량              | 가중치(점수) | 설명                                                  |
|----------------|---------------------|--------------|--------------------------------------------------------|
| total_calories | 2,500 kcal          | 30점         | 하루 권장 칼로리 근사치. 칼로리 과다 시 위험 증가      |
| total_fat      | 70 g                | 25점         | 지방 과다 섭취 시 비만·대사이상 위험 증가              |
| total_sugar    | 50 g                | 20점         | 당류 섭취 과다 시 인슐린 저항성·비만 관련 위험 증가    |
| total_sodium   | 2,000 mg            | 25점         | 나트륨 과다 섭취 시 복부비만·비만 위험 증가            |

합계 = **100점 만점**,  
즉, 사용자의 위험도는 항상 **0점 이상 ~ 100점 이하** 범위 내에 분포하도록 설계되었습니다.

### 의료·역학적 근거

- 국내외 연구에서 **나트륨 섭취가 비만 지표(체지방률, 복부비만 등)와 유의미한 양의 상관관계**를 보였습니다. :contentReference[oaicite:0]{index=0}  
- 또한, **지방 (total_fat) 및 당류 (total_sugar) 섭취 증가**가 비만 또는 이상지질혈증과 연계되어 있다는 한국인 대상 체계적 고찰이 존재합니다. :contentReference[oaicite:1]{index=1}  
- 본 공식은 이러한 역학적 근거를 바탕으로 “칼로리 + 지방 + 당류 + 나트륨” 네 가지 핵심 요소에 집중하여 설계된 **팀 내부 모델**입니다.  
- 따라서 이 모델의 값은 **임상시험에서 직접 제시된 공식**은 아니며, 여러 연구 근거를 종합하여 우리 서비스 목적에 맞게 설계된 점수를 의미합니다.

---

## 4. 학습 데이터 구성 및 입력 변수

### 입력 (X) 변수
- total_calories  
- total_fat  
- total_sugar  
- total_sodium  
- carb_ratio (탄수화물 비율)  
- protein_ratio (단백질 비율)  

### 출력 (y) 변수
- 상단 공식으로 계산된 **risk_score (0~100)**  

### 특징
- BMI(체질량지수)는 이 모델에서 **사실상 제외**하거나 영향이 매우 낮도록 설계됨.  
- 모델은 “오늘 먹은 음식”만으로 위험도를 판단하는 구조입니다.

---

## 5. 머신러닝 모델 설계

- 모델 타입: **RandomForestRegressor** (또는 유사 회귀모델)  
- 목표: 연속형 점수(0~100) 출력  
- 이유:
  - 이진 분류(Logistic) → 0 또는 1만 나올 위험 존재 → **부적합**  
  - 회귀모델 → 0~100 사이 다양한 값 출력 가능 → **적합**  
- 가중치 반영: 학습 데이터에 의해 “칼로리 ↑, 지방 ↑, 당류 ↑, 나트륨 ↑ → 위험도 ↑” 패턴이 자연스럽게 학습되도록 설계  

---

## 6. 학습 코드 예시 (`train.py`)

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load nutrition dataset
df = pd.read_csv("nutrition.csv")

# Features
X = df[[
    "total_calories",
    "total_fat",
    "total_sugar",
    "total_sodium",
    "carb_ratio",
    "protein_ratio"
]]

# Risk Score 생성
df["risk_score"] = (
    (df["total_calories"] / 2500 * 30) +
    (df["total_fat"] / 70 * 25) +
    (df["total_sugar"] / 50 * 20) +
    (df["total_sodium"] / 2000 * 25)
)

y = df["risk_score"]

# 모델 훈련
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# 모델 저장
joblib.dump(model, "obesity_model_v2.pkl")

print("Model training complete: obesity_model_v2.pkl saved.")
```

---

## 7. FastAPI 예측 API 설계 (`main.py` 및 `db.py` 포함)

### `db.py` (DB 연결 예시)
```python
import pymysql

def get_connection():
    return pymysql.connect(
        host="YOUR_HOST",
        user="YOUR_USER",
        password="YOUR_PASSWORD",
        database="YOUR_DB",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
```

### `main.py`
```python
from fastapi import FastAPI
import numpy as np
import joblib
import os
from db import get_connection

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "obesity_model_v2.pkl"))

def get_today_nutrition(user_id: int):
    conn = get_connection()
    cur = conn.cursor()
    sql = """
        SELECT
            IFNULL(SUM(CALORIES_KCAL),0) AS total_calories,
            IFNULL(SUM(FATS_G),0) AS total_fat,
            IFNULL(SUM(SUGAR_G),0) AS total_sugar,
            IFNULL(SUM(SODIUM_MG),0) AS total_sodium,
            IFNULL(SUM(CARBS_G),0) AS total_carbs,
            IFNULL(SUM(PROTEIN_G),0) AS total_protein
        FROM MEAL_ENTRY
        WHERE USER_ID = %s
          AND DATE(EAT_TIME) = CURDATE()
    """
    cur.execute(sql, (user_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    total_cal = result["total_calories"]
    carbs = result["total_carbs"]
    protein = result["total_protein"]
    fat = result["total_fat"]
    if total_cal > 0:
        carb_ratio = carbs * 4 / total_cal * 100
        protein_ratio = protein * 4 / total_cal * 100
    else:
        carb_ratio = protein_ratio = 0

    return {
        "total_calories": total_cal,
        "total_fat": fat,
        "total_sugar": result["total_sugar"],
        "total_sodium": result["total_sodium"],
        "carb_ratio": carb_ratio,
        "protein_ratio": protein_ratio
    }

@app.get("/predict-risk/{user_id}")
def predict_risk(user_id: int):
    data = get_today_nutrition(user_id)
    features = [[
        data["total_calories"],
        data["total_fat"],
        data["total_sugar"],
        data["total_sodium"],
        data["carb_ratio"],
        data["protein_ratio"]
    ]]
    pred = model.predict(features)[0]
    pred = max(0, min(100, pred))
    return {
        "user_id": user_id,
        "risk_score": round(pred, 2)
    }
```

---

## 8. Spring 연동 코드

```java
public int getObesityPercent(int userId) {
    String url = "http://YOUR_API_HOST:8001/predict-risk/" + userId;
    Map<String, Object> result = restTemplate.getForEntity(url, Map.class).getBody();

    double score = 0.0;
    if (result != null && result.get("risk_score") != null) {
        score = Double.parseDouble(result.get("risk_score").toString());
    }

    return (int) Math.round(score);
}
```

이 구조를 통해 Spring 애플리케이션은 FastAPI 서버에 요청을 보내 **연속형 위험도 점수(risk_score)** 를 받아올 수 있습니다.

---

## 9. “0~100” 점수의 의미

- **0점**: 오늘의 식단 기준으로 보면 위험요인이 거의 없음  
- **100점**: 오늘의 식단 기준으로 보면 매우 위험 수준에 근접함  
- **중간값(예: 35.6점, 72.4점 등)**: 
  - 측정된 값이 “보통보다 위험함” 또는 “위험 수준에 가깝다”는 의미  
  - 즉, 이 점수는 “오늘 내가 먹은 음식이 얼마나 비만 위험을 높일 수 있는가”를 정량적으로 보여줍니다  
- 이 점수는 **비만 여부(지속적 상태)**가 아니라, **오늘 식단이 얼마나 즉각적으로 위험도를 높일 가능성이 있는가**를 나타내는 지표입니다  

---

## 10. 모델 설계 요약

- BMI 영향은 **사실상 제거**됨 → 입력 변수에서 제외하거나 가중치를 낮춤  
- 오늘 먹은 음식만으로 위험도를 산출  
- 칼로리, 지방(fat), 당(sugar), 나트륨(sodium)이 가장 큰 영향  
- 탄수화물 비율(carb_ratio)과 단백질 비율(protein_ratio)은 영향이 낮음  
- 모델 출력은 **연속형 0~100 점수**  
- 서비스 구조: FastAPI(모델 서버) + Spring(웹 서버) + 프론트엔드  
- 사용자 경험: “내 식단이 오늘 얼마나 위험했는지”를 한눈에 확인  

---

## 11. 한계 및 유의사항

- 이 공식은 **연구 논문에서 동일한 수식으로 제시된 것은 아니며**, 내부 설계 모델입니다  
- 역학적 연구는 주로 **상관관계**를 보여주며, 인과관계 확정은 어려움 :contentReference[oaicite:2]{index=2}  
- 입력 데이터 품질(영양소 집계 등)에 따라 예측 값이 달라질 수 있음  
- 모델이 **오늘 먹은 음식만** 반영하므로, 운동·기초대사량·체중변화 등의 요소는 고려되지 않음  
- 위험도는 “즉시 위험 가능성”을 보여주며, 장기적 비만 상태나 질환 위험을 직접 나타내는 것은 아님  

---

## 12. 결론

본 모델은 SaveUs 서비스에서 사용되는 **새로운 비만 위험도 분석 표준**입니다.  
사용자가 오늘 먹은 음식만으로 **즉시 계산되는 건강 위험도(0~100 점수)**를 제공합니다.
