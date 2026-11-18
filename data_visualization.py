import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def main():
    df = pd.read_csv("./datasets/raw/RT_IOT2022.csv")

    # Attack types class distribution
    plt.figure(figsize=(8, 8))
    df['Attack_type'].value_counts().plot(kind='bar', edgecolor='black')
    plt.title('Class Distribution — Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Protocols class distribution
    plt.figure(figsize=(8, 8))
    df['proto'].value_counts().plot(kind='bar', edgecolor='black')
    plt.title('Class Distribution — Protocols')
    plt.xlabel('Protocols')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Box plot Attack type v/s ARP Poisoning
    arp_data = df[df['Attack_type'] == 'ARP_poisioning']
    arp_data.boxplot(column='flow_duration', by='Attack_type', figsize=(6, 6))
    plt.title('flow_duration — ARP Poisoning')
    plt.suptitle('')
    plt.xlabel('Attack Type')
    plt.ylabel('Flow Duration (s)')
    plt.ylim(0, df['flow_duration'].quantile(0.95))  # cut off top 5% outliers
    plt.tight_layout()
    plt.show()

    # Box plot Attack type v/s Thing Speak
    arp_data = df[df['Attack_type'] == 'Thing_Speak']
    arp_data.boxplot(column='flow_duration', by='Attack_type', figsize=(6, 6))
    plt.title('flow_duration — Thing Speak')
    plt.suptitle('')
    plt.xlabel('Attack Type')
    plt.ylabel('Flow Duration (s)')
    plt.ylim(0, df['flow_duration'].quantile(0.95))  # cut off top 5% outliers
    plt.tight_layout()
    plt.show()


    # Box plot Attack type v/s Metasploit Brute Force SSH
    arp_data = df[df['Attack_type'] == 'Metasploit_Brute_Force_SSH']
    arp_data.boxplot(column='flow_duration', by='Attack_type', figsize=(6, 6))
    plt.title('flow_duration — Metasploit Brute Force SSH')
    plt.suptitle('')
    plt.xlabel('Attack Type')
    plt.ylabel('Flow Duration (s)')
    plt.ylim(0, df['flow_duration'].quantile(0.95))  # cut off top 5% outliers
    plt.tight_layout()
    plt.show()

    print(df.select_dtypes(include=['object']).columns.tolist())

    print("proto unique values:", df['proto'].unique())
    print("service unique values:", df['service'].unique())


if __name__  == "__main__":
    main()      

    