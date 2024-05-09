"""
Takes output stream from given file, and formats it. Extracts key statistics and plots it.  
- generates histogram for top called functions & cumulative time. Takes in a dictionary from profiles to the video source 
"""


import pstats 
from pstats import SortKey
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

#Settings
SEABORN = False 
MATPLOTLIB = False

def profile_to_dataframe(profile_file, sort_key='cumulative'):
    # Load the profiling statistics
    stats = pstats.Stats(profile_file)
    
    # Sort the statistics by the specified key
    stats.sort_stats(sort_key)
    
    # Extract data from stats to a list
    data_list = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line_number, function_name = func
        data_list.append({
            'filename': filename,
            'line_number': line_number,
            'function_name': function_name,
            'total_calls': cc,
            'primitive_calls': nc,
            'total_time': tt,
            'cumulative_time': ct
        })
    
    # Create a DataFrame from the list
    df = pd.DataFrame(data_list)
    df.sort_values(by='cumulative_time', ascending=False, inplace=True)
    return df

"""Plot the top n functions by cumulative time"""
def plot_profile(df, n=10):
    # Plotting the top 10 functions with the highest cumulative time
    top_df = df.head(10)
    fig, ax = plt.subplots()
    top_df.plot(kind='barh', x='function_name', y='cumulative_time', ax=ax)
    ax.set_xlabel('Cumulative Time (seconds)')
    ax.set_title('Top 10 Functions by Cumulative Time')
    plt.show()

def seaborn_plot(df, n=10): 
    top_df = df.head(5)
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.displot(data=top_df, x='function_name', y='cumulative_time')
    plt.show()

if __name__ == '__main__':
    # Load the profile data into a DataFrame
    filepath = "/Users/charlesgordon/Desktop/Research/290/final_project_repo/eval_pi5/profiles_monkeydog_5frames.prof"
    stats = pstats.Stats(filepath)
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(20)
    #df = profile_to_dataframe(filepath, 'cumulative')
    if SEABORN: 
        seaborn_plot(df)
    if MATPLOTLIB: 
        plot_profile(df)
    


