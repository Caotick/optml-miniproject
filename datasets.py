from __future__ import print_function, division
import pandas as pd
from torch.utils.data import Dataset
import torch


class PIMADataset(Dataset):
    """
    PIMA Indian Diabetes Database
    """

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Standardising
        X = df.drop('Outcome', axis=1)
        X = (X - X.mean()) / X.std()
        X['Outcome'] = df['Outcome']

        self.data = X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO cast to tensor ?
        features = torch.tensor(self.data.iloc[idx].drop('Outcome').values, dtype=torch.float32)
        target = int(self.data.iloc[idx]['Outcome'])
        return features, target


class HousingDataset(Dataset):
    """
    California Housing Prices
    """
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Standardising
        X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
        # TODO standardise lat lon ?
        X = (X - X.mean()) / X.std()
        X['median_house_value'] = df['median_house_value']

        self.data = X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx].drop('median_house_value').values, dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx]['median_house_value'], dtype=torch.float32)
        return features, target
