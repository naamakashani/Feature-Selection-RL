import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_compare():
    # Example data for the existing columns (PCA, IG, MI, CST, CFS)
    data = {
        'IG': np.random.normal(loc=63, scale=8, size=100),
        'MI': np.random.normal(loc=64, scale=8.2, size=100),
        'CST': np.random.normal(loc=62, scale=8.1, size=100),
        'CFS': np.random.normal(loc=63, scale=7.8, size=100),
        'PFS': np.random.normal(loc=70, scale=5, size=100),
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Number of features selected by each method
    feature_counts = {
        'IG': 15,
        'MI': 15,
        'CST': 15,
        'CFS': 15,
        'PFS': 9.27,
    }

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)

    # Customize plot
    plt.xticks(range(len(df.columns)), [f'{col}\n{feature_counts[col]} features' for col in df.columns], rotation=0,
               size=12)
    plt.ylabel('Mean Acc. (%)', size=12)
    plt.yticks(size=12)
    plt.title('Accuracy Comparison of Different Feature Selection Methods')
    plt.ylim(40, 100)
    plt.grid(True)

    # Adding median annotations
    for i, col in enumerate(df.columns):
        median_val = df[col].median()
        plt.text(i, median_val, f'{median_val:.2f}', horizontalalignment='center', color='black', weight='semibold',
                 size=12)

    plt.tight_layout()

    plt.show()


def plot_table():
    import matplotlib.pyplot as plt
    import numpy as np

    # Data from the table
    configurations = [
        ("Random", "Uniform", "MSE", "Pre-training only", "Features vector", 0.70, 6.4, 415),
        ("Difference", "Uniform", "MSE", "Pre-training only", "Features vector", 0.72, 6.4, 250),
        ("Difference", "Priority", "MSE", "Pre-training only", "Features vector", 0.71, 6, 231),
        ("Difference", "Priority", "Huber", "Pre-training only", "Features vector", 0.75, 5.9, 221),
        ("Difference", "Priority", "Huber", "Robust Guesser", "Features vector", 0.74, 5.6, 240),
        ("Difference", "Priority", "Huber", "Every 1000 iterations", "Features vector", 0.72, 8.6, 300),
        ("Difference", "Priority", "Huber", "No-pre-training", "LSTM state", 0.72, 8.1, 350)
    ]

    # Extracting data for each metric
    rewards = [config[0] for config in configurations]
    replay_memory = [config[1] for config in configurations]
    loss = [config[2] for config in configurations]
    guesser_train = [config[3] for config in configurations]
    state_representation = [config[4] for config in configurations]
    accuracies = [config[5] for config in configurations]
    feature_counts = [config[6] for config in configurations]
    epoch_counts = [config[7] for config in configurations]

    # Unique labels for x-axis ticks
    labels = [f"{rewards[i]} | {replay_memory[i]} | {loss[i]} | {guesser_train[i]} | {state_representation[i]}" for i in
              range(len(configurations))]

    # Width of each bar
    bar_width = 0.25

    # Positions of bars on x-axis
    r1 = np.arange(len(configurations))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Plotting
    plt.figure(figsize=(14, 8))

    plt.bar(r1, accuracies, color='b', width=bar_width, edgecolor='grey', label='Accuracy')
    plt.bar(r2, feature_counts, color='r', width=bar_width, edgecolor='grey', label='Feature Count')
    plt.bar(r3, epoch_counts, color='g', width=bar_width, edgecolor='grey', label='Epoch Count')

    # Adding labels and ticks
    plt.xlabel('Configurations', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(configurations))], labels, rotation=90)
    plt.title('Comparison of Metrics Across Configurations', fontsize=16, fontweight='bold')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
 plot_table()