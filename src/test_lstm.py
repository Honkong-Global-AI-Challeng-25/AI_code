import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
import os
from torch import nn

# 1. 시간-온도 feature 정의
def temp_time_effect(dt, temperature, max_temp=35):
    hour_decimal = dt.hour + dt.minute / 60
    time_sym_weight = np.cos(np.pi * (hour_decimal - 12) / 12)
    norm_temp = temperature / max_temp
    return time_sym_weight * norm_temp


# 모델 및 스케일러 로드
def load_model(model_path, scaler_X_path, scaler_y_path):
    # 장치 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 구성 정보 가져오기
    config = checkpoint['model_config']
    
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.4):
            super(LSTMModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_prob if num_layers > 1 else 0
            )
            
            self.dropout = nn.Dropout(dropout_prob)
            self.fc = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.fc(out)
            
            return out
    
    # 모델 인스턴스 생성
    model = LSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size'],
        dropout_prob=config['dropout_prob']
    ).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 스케일러 로드
    scaler_X = joblib.load(scaler_X_path)  # 실제 경로로 변경
    scaler_y = joblib.load(scaler_y_path)  # 실제 경로로 변경
    
    return model, scaler_X, scaler_y, device


# 테스트 데이터 읽기 및 처리
def process_test_data(test_file_path):
    # CSV 파일 읽기
    test_df = pd.read_csv(test_file_path)
    
    # prediction_time 열을 datetime 형식으로 변환
    #test_df['prediction_time'] = pd.to_datetime(test_df['prediction_time'], dayfirst=True)
    test_df['prediction_time'] = pd.to_datetime(test_df['prediction_time'], format="%Y-%m-%d %H:%M:%S")
    # 시간 특성 추가
    test_df['hour'] = test_df['prediction_time'].dt.hour
    test_df['day_of_week'] = test_df['prediction_time'].dt.dayofweek
    test_df['month'] = test_df['prediction_time'].dt.month
    test_df['day'] = test_df['prediction_time'].dt.day
    test_df['temp_time_feature'] = [
        temp_time_effect(dt, temp) for dt, temp in zip(test_df['prediction_time'], test_df['mean_temperature'])
    ]
    
    return test_df


"""
# 시간 데이터로 예측 수행
def predict_power_consumption(model, test_df, scaler_X, scaler_y, device):
    model.eval()  # 평가 모드 설정
    
    # 시간 특성 추출
    #time_features = test_df[['hour', 'day_of_week', 'month', 'day']].values
    time_features = test_df[['hour', 'day_of_week', 'month', 'day', 'temp_time_feature', 'mean_temperature']].values
    # 전체 입력 특성 준비 (나머지는 0으로 채움)
    # 모든 특성을 0으로 초기화
    full_features = np.zeros((len(test_df), scaler_X.n_features_in_))
    
    # 시간 특성 채우기 - 실제 특성 순서에 맞게 조정 필요
    # 예: 시간 특성이 마지막 4개 열이라고 가정
    feature_indices = list(range(scaler_X.n_features_in_ - 6, scaler_X.n_features_in_))
    full_features[:, feature_indices] = time_features
    
    # 입력 정규화
    normalized_features = scaler_X.transform(full_features)
    
    with torch.no_grad():
        # 입력을 텐서로 변환
        # LSTM은 [batch_size, seq_length, input_size] 형태의 입력을 기대
        # 각 시간점에 대해 독립적으로 예측하므로 seq_length=1로 설정
        input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(1).to(device)
        
        # 예측
        predictions = model(input_tensor).cpu().numpy()
        
        # 예측값 역정규화
        predictions_original = scaler_y.inverse_transform(predictions)
    
    return predictions_original.flatten()


# 결과를 원하는 형식으로 저장
def save_predictions(test_df, predictions, output_file='predicted_power_loads_nrmse_add_temperature.csv'):
    # 예측값 반올림 (정수로)
    
    # 시간과 예측값 사이에 쉼표 구분자 추가하여 CSV 생성
    rows = []
    for i, dt in enumerate(test_df['prediction_time']):
        # 날짜/시간 형식 설정 (앞의 0 제거)
        day = dt.day
        month = dt.month
        year = dt.year
        hour = dt.hour
        minute = dt.minute
        
        # 분이 0인 경우 00으로 표시
        if minute == 0:
            minute_str = "00"
        else:
            minute_str = str(minute)
        
        # 시간 형식 생성
        time_str = f"{day}/{month}/{year} {hour}:{minute_str}"
        
        # 행 추가 (시간,예측값)
        rows.append(f"{time_str},{predictions[i]}")
    
    # 행들을 줄바꿈으로 구분하여 파일에 저장 (헤더는 포함하지 않음)
    with open(output_file, 'w') as f:
        f.write("\n".join(rows))
    
    print(f"예측 결과가 {output_file}에 저장되었습니다.")
    
    return rows
"""

def predict_power_consumption(model, test_df, scaler_X, scaler_y, device):
    model.eval()

    # 전체 feature 이름: 학습 시와 동일한 순서
    features = [
        'CHR-01-KW', 'CHR-01-CHWSWT', 'CHR-01-CHWRWT', 'CHR-01-CHWFWR',
        'CHR-02-KW', 'CHR-02-CHWSWT', 'CHR-02-CHWRWT', 'CHR-02-CHWFWR',
        'CHR-03-KW', 'CHR-03-CHWSWT', 'CHR-03-CHWRWT', 'CHR-03-CHWFWR',
        'mean_temperature', 'hour', 'day_of_week', 'month', 'day', 'temp_time_feature'
    ]

    # 테스트 데이터셋에서는 센서값(전력, 유량 등)은 예측 시점에 알 수 없으므로 0으로 채움
    X_input = np.zeros((len(test_df), len(features)))

    # 시간/온도 관련 실제값 삽입
    X_input[:, features.index('mean_temperature')] = test_df['mean_temperature']
    X_input[:, features.index('hour')] = test_df['hour']
    X_input[:, features.index('day_of_week')] = test_df['day_of_week']
    X_input[:, features.index('month')] = test_df['month']
    X_input[:, features.index('day')] = test_df['day']
    X_input[:, features.index('temp_time_feature')] = test_df['temp_time_feature']

    # 정규화
    X_scaled = scaler_X.transform(X_input)

    # 특성별 가중치 부여 (학습 시와 일치)
    weights = {'mean_temperature': 3.0, 'hour': 3.0, 'day': 1.0, 'temp_time_feature': 3.0}
    for feat, w in weights.items():
        X_scaled[:, features.index(feat)] *= w

    # 모델 입력: [batch_size, seq_len=1, input_size]
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        predictions = model(input_tensor).cpu().numpy()
        predictions_original = scaler_y.inverse_transform(predictions)

    return predictions_original.flatten()


def save_predictions(test_df, predictions, output_file='predicted_power_loads_nrmse_add_temperature_add_weight.csv'):
    with open(output_file, 'w') as f:
        f.write("prediction_time,predicted_load\n")  # 헤더 추가
        for i, dt in enumerate(test_df['prediction_time']):
            timestamp_str = dt.strftime('%-d/%-m/%Y %-H:%M')  # 예: 1/1/2024 0:00
            f.write(f"{timestamp_str},{predictions[i]:.10f}\n")
    print(f"✅ 예측 결과가 {output_file}에 저장되었습니다.")







test_file_path = '/home2/escho/global_25_version2/pure-lstm_eunsoo_version/pure-lstm/preprocess/test_with_temperature.csv'  # 테스트 데이터 파일 경로
output_file = 'predicted_power_loads_2nd_add_temperature_add_weight_2.csv'  # 결과 저장 파일 경로
model, scaler_X, scaler_y, device = load_model("chiller_lstm_model_second_add_temperature_add_weight.pth", "scaler_X_second_add_temperature_add_weight.pkl", "scaler_y_second_add_temperature_add_weight.pkl")

# 테스트 데이터 처리
test_df = process_test_data(test_file_path)

# 예측 수행
predictions = predict_power_consumption(model, test_df, scaler_X, scaler_y, device)

# 결과 저장
save_predictions(test_df, predictions, output_file)
