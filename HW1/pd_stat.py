import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('AB_NYC_2019.csv')
    mean = data.price.mean()
    var = data.price.var()
    print(f'pd mean: {mean}')
    print(f'pd var: {var}')
