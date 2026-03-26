import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ==========================
#  قراءة البيانات
# ==========================
data = pd.read_csv("heart_data.csv")
print("Dataset Preview:")
print(data.head())

# ==========================
#  تحليل معدل نبض القلب واكتشاف القيم العالية
# ==========================
avg_hr = data["HeartRate"].mean()
print(f"\nAverage Heart Rate: {avg_hr:.1f}")

high_hr = data[data["HeartRate"] > 100]
print("\nHigh Heart Rate Records:")
print(high_hr)

# ==========================
# رسم بياني لتتبع نبض القلب
# ==========================
plt.plot(data["Day"], data["HeartRate"], marker='o')
plt.title("Heart Rate Trend")
plt.xlabel("Day")
plt.ylabel("Heart Rate")
plt.grid(True)
plt.show()

# ==========================
#  Insight ذكي 
# ==========================
if data["SleepHours"].mean() < 6.5:
    print("\nInsight: Average sleep is low. Heart rate may increase due to insufficient rest.")
else:
    print("\nInsight: Sleep hours are adequate. Heart rate is likely normal.")

# ==========================
#  Machine Learning Prediction 
# ==========================
# تحضير البيانات
X = data["SleepHours"].values.reshape(-1, 1)  # feature: SleepHours
y = data["HeartRate"].values                 # target: HeartRate

# إنشاء النموذج وتدريبه
model = LinearRegression()
model.fit(X, y)

# التوقع لساعات نوم مختلفة
sleep_hours_test = np.array([5, 6, 7]).reshape(-1, 1)
predicted_hr = model.predict(sleep_hours_test)

print("\nPredicted Heart Rates for Sleep Hours 5, 6, 7:")
for sh, hr in zip(sleep_hours_test.flatten(), predicted_hr):
    print(f"{sh} hours sleep → predicted heart rate: {hr:.1f}")
