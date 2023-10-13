import numpy as np
import pandas as pd


def load_plyl_data():
    # 看见dataFrame的所有行，不然会省略
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    rawData = pd.read_csv(filepath_or_buffer='../data/BAI246_CAL_GR_AC_SP.csv', header=0, encoding='utf-8',
                          dtype=np.float32, on_bad_lines='skip')
    return rawData


if __name__ == '__main__':
    pass
