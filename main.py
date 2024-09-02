import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 데이터 로드 (사용자 데이터로 대체)
data = pd.read_csv('data.csv')  # 실제 데이터 파일 경로로 변경

# 독립 변수와 종속 변수 정의
X = data[["HRD학점계", "MSC학점계", "교양학점계", "전공학점계", "총 평점", "전체석차", "평균수강학점", "학사경고여부", "복수전공이수여부", "부전공이수여부", "현장실습기간구분", "입학장학명", "성적우수장학명", "출석점수"]]
y = data['취업결과']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 학습
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 결과 평가
print(classification_report(y_test, y_pred))

# 회귀 계수 출력
coefficients = pd.DataFrame(model.coef_, columns=X.columns, index=model.classes_)
print("\n회귀 계수:")
print(coefficients)

# 새로운 데이터로 예측하기
new_data = [[1.2, 3.4, 2.1, 3.5, 4.0, 120, 3.0, 0, 1, 0, 1, 0, 1, 90]]  # 새로운 데이터 입력
new_data = scaler.transform(new_data)  # 표준화
prediction = model.predict(new_data)
print("\n새로운 데이터의 예측 결과:")
print(prediction)

