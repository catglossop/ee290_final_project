"""Generating plots for multiobjecttracking profile"""

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data from a CSV file and plot a histogram
def plot_histogram(data):
    # Load the dataset from a CSV fil

    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid", palette="pastel")

    # Create a histogram plot
    sns.histplot(data=data, x="Function", y="Cumulative Time (Percentage of Execution)", hue="Video", multiple="dodge", shrink=.8, common_bins=True)

    # Enhance the plot with titles and labels
    plt.title('Profile of Multi-Object Tracking')
   # plt.xlabel("")
    #plt.ylabel('Frequency')

    # Show the plot
    plt.show()


if __name__ == '__main__':
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

    #filepath = "/Users/charlesgordon/Downloads/Profile_data2.csv"
    plot_histogram(data)

