import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="dataset/News_Category_Dataset_v3_balanced.json")

args = parser.parse_args()


def main():
    df_news_data = pd.read_json(args.data_path, lines = True)
    print(df_news_data)

if __name__ == '__main__':
    main()