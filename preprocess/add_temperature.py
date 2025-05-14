import pandas as pd
import os
# 1. 평균 기온 데이터 불러오기
# 1. 평균 기온 파일 불러오기
temp_df = pd.read_csv("C:/Users/82104/Desktop/홍콩공모전/pure-lstm/data/average_temperature/daily_KP_MEANHKHI_ALL_2014_2025.csv", skiprows=2, encoding="utf-8")

# 2. 컬럼명 변경
temp_df = temp_df.rename(columns={
    "年/Year": "year",
    "月/Month": "month",
    "日/Day": "day",
    "數值/Value": "mean_temperature"
})

# ❗ 비정상 날짜 필터링 처리
temp_df["year"] = pd.to_numeric(temp_df["year"], errors="coerce")
temp_df["month"] = pd.to_numeric(temp_df["month"], errors="coerce")
temp_df["day"] = pd.to_numeric(temp_df["day"], errors="coerce")
temp_df = temp_df.dropna(subset=["year", "month", "day"])
temp_df[["year", "month", "day"]] = temp_df[["year", "month", "day"]].astype(int)


# 3. 날짜 조합
temp_df["date"] = pd.to_datetime(temp_df[["year", "month", "day"]])

# 4. 필요한 컬럼만 정리
temp_df = temp_df[["date", "mean_temperature"]]

# 2. 센서 데이터 불러오기
data_df = pd.read_csv("C:/Users/82104/Desktop/홍콩공모전/pure-lstm/data/raw-data/test.csv")

# 날짜 변환 (day/month/year 순서)
#data_df["record_timestamp"] = pd.to_datetime(data_df["record_timestamp"], dayfirst=True)
data_df["prediction_time"] = pd.to_datetime(data_df["prediction_time"], dayfirst=True)
# 날짜만 추출하여 join 기준으로 사용
data_df["date"] = data_df["prediction_time"].dt.normalize()


# 3. 날짜 기준 merge
merged_df = pd.merge(data_df, temp_df, how="left", on="date")

# 4. 정리
merged_df.drop(columns=["date"], inplace=True)

merged_df.to_csv("test_with_temperature.csv", index=False)

