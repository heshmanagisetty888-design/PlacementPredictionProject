import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel("../Data/placement_predict_50k_Dataset.xlsx")
plt.figure(figsize=(6,4))
plt.hist(df["CGPA"], bins=10,edgecolor="black")
plt.title("Histogram of CGPA")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
sns.boxplot(
    x="Placement",
    y="CGPA",
    data=df,
)
plt.title("Box Plot of CGPA")
plt.show()
plt.show()