import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('HR_Analytics.csv')
print('dataset shape:',df.shape)
print('first 5 rows:',df.head())
print('missing values:',df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print('Attrition count:',df["Attrition"].value_counts())
print(df.info())
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)
plt.title("Employee Attrition by Department")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)
plt.title("Employee Attrition by Overtime")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)
plt.title("Monthly Income vs Employee Attrition")
plt.tight_layout()
plt.show()

df["SalaryBand"] = pd.cut(
    df["MonthlyIncome"],
    bins=[0, 3000, 6000, 10000, float("inf")],
    labels=["Low", "Medium", "High", "Very High"]
)

plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="SalaryBand",
    hue="Attrition"
)
plt.title("Employee Attrition by Salary Band")
plt.tight_layout()
plt.show()
