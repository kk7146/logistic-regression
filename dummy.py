import pandas as pd

# Load the dataset
df = pd.read_csv('updated_dummy_data_500_final_scholarships.csv')

# 출석점수 계산
df['출석점수'] = 100 - (df['지각일수'] * 2 + df['조퇴일수'])

# 지각일수와 조퇴일수 열 제거
df = df.drop(columns=['지각일수', '조퇴일수'])

# Function to calculate the scores for each company type based on the given variables
def calculate_employment_scores(row):
    weights = {
        1: 0,  # 대기업
        2: 0,  # 공기업
        3: 0,  # 중견기업
        4: 0   # 중소기업
    }
    weights[1] = 5
    weights[2] = 6
    weights[3] = 13
    weights[4] = 16

    # 대기업(1)과 공기업(2) 확률을 학점에 따라 제한
    if row['총 평점'] >= 4.0:
        weights[1] += row['총 평점'] * 2.5 - row['전체석차'] * 0.03
        weights[2] += row['총 평점'] * 1.8 - row['전체석차'] * 0.012  # 공기업 가중치 감소
    else:
        weights[1] -= 100  # 학점이 4.0 미만이면 대기업 선택을 제한
    
    # 공기업도 학점이 높을수록 가중치 증가
    if row['총 평점'] >= 3.5:
        weights[2] += row['총 평점'] * 0.4 - row['전체석차'] * 0.01
    else:
        weights[2] -= 5  # 학점이 3.5 미만이면 공기업 선택을 제한
    
    # 중견기업도 학점이 3.0 이상이면 가중치 증가
    if row['총 평점'] >= 3.5:
        weights[3] += 3


    # 출석점수는 모든 기업에 긍정적 영향
    weights[1] += row['출석점수'] * 0.3
    weights[2] += row['출석점수'] * 0.3
    weights[3] += row['출석점수'] * 0.3
    weights[4] += row['출석점수'] * 0.3
    
    # 학사경고 여부는 공기업과 대기업 취업에 부정적 영향
    if row['학사경고여부'] == 1:
        weights[1] -= 10.0
        weights[2] -= 3.0
    
    # 복수전공과 부전공 여부는 대기업 및 공기업에 긍정적 영향
    if row['복수전공이수여부'] == 1:
        weights[1] += 1.0
        weights[2] += 0.6
    if row['부전공이수여부'] == 1:
        weights[1] += 0.8
        weights[2] += 0.4
    
    # 현장실습기간 구분: 장기실습일수록 대기업 및 공기업 선호 확률 증가
    if row['현장실습기간구분'] == 1:
        weights[1] += 1.2
        weights[2] += 1.0  # 공기업에 대한 장기실습 가중치 감소
    
    # 성적 우수 장학명과 입학 장학명: 장학금이 좋을수록 대기업 및 공기업 선호 확률 증가
    weights[1] += (5 - row['성적우수장학명']) * 1.5
    weights[2] += (5 - row['성적우수장학명']) * 1.2  # 공기업 가중치 감소
    weights[1] += (8 - row['입학장학명']) * 1.0
    weights[2] += (8 - row['입학장학명']) * 0.8  # 공기업 가중치 감소
    
    return weights

# Apply the function to calculate the employment scores for each student
df['대기업_가중치'] = df.apply(lambda row: calculate_employment_scores(row)[1], axis=1)
df['공기업_가중치'] = df.apply(lambda row: calculate_employment_scores(row)[2], axis=1)
df['중견기업_가중치'] = df.apply(lambda row: calculate_employment_scores(row)[3], axis=1)
df['중소기업_가중치'] = df.apply(lambda row: calculate_employment_scores(row)[4], axis=1)

# Determine the predicted employment location and its probability
df['취업예상지'] = df[['대기업_가중치', '공기업_가중치', '중견기업_가중치', '중소기업_가중치']].idxmax(axis=1)
df['취업예상지'] = df['취업예상지'].map({'대기업_가중치': 1, '공기업_가중치': 2, '중견기업_가중치': 3, '중소기업_가중치': 4})
df['취업확률'] = df[['대기업_가중치', '공기업_가중치', '중견기업_가중치', '중소기업_가중치']].max(axis=1)

# Save the updated DataFrame to a new CSV file
output_file_path_with_employment_v10 = 'C:/Users/dsino/Desktop/updated_dummy_data_500_with_employment_v10.csv'
df.to_csv(output_file_path_with_employment_v10, index=False)

print("Updated file saved at:", output_file_path_with_employment_v10)
