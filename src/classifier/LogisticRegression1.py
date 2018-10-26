from DataProcessing import load_data
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

PARAMS = [{'penalty': ["l1", "l2"], 'C': [4, 2, 1.5, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]}]

data = load_data()
print(list(data.columns))
