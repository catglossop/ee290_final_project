import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data given by the user
data = {
    "Function": ["Feature Detection", "Homography", "imRead", "imresize", "Descriptor Matching",
                 "Feature Detection", "Homography", "imRead", "imresize", "Descriptor Matching",
                 "Feature Detection", "Homography", "imRead", "imresize", "Descriptor Matching",
                 "Feature Detection", "Homography", "imRead", "imresize", "Descriptor Matching"],
    "Video": ["Girl", "Girl", "Girl", "Girl", "Girl",
              "Frog", "Frog", "Frog", "Frog", "Frog",
              "Drift", "Drift", "Drift", "Drift", "Drift",
              "Bird of Paradise", "Bird of Paradise", "Bird of Paradise", "Bird of Paradise", "Bird of Paradise"],
    "Cumulative Time (Percentage of Execution)": [53.30396476, 4.933920705, 14.97797357, 4.845814978, 5.242290749,
                                                  5.855855856, 0.2815315315, 45.83333333, 8.220720721, 0.2815315315,
                                                  17.15976331, 1.174978867, 30.43110735, 30.43110735, 2.113271344,
                                                  10.51677244, 2.719854941, 51.85856754, 0.9972801451, 2.719854941]
}

data['Number of Calls'] = [10, 9, 21, 42, 9,
                           10, 9, 279, 558, 9,
                           20, 18, 74, 222, 18,
                           10, 9, 98, 196, 9]
# Creating DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(data=df, x='Function', y='Number of Calls', hue='Video')
plt.title('Calls by Function and Video')
plt.xticks(rotation=45)
plt.xlabel('Function')
plt.ylabel('Number of Calls')
plt.grid(True)
plt.legend(title='Video', loc='upper right')
plt.tight_layout()
plt.show()
