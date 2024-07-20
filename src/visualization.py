import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df, save_path='heatmap.png'):
    plt.figure(figsize=(15, 15))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="PuBuGn")  # Select only numeric columns
    plt.savefig(save_path)
    plt.close()

def plot_count(df, column, hue=None, save_path=None):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, hue=hue, data=df)
    if save_path:
        plt.savefig(save_path)
    plt.close()
