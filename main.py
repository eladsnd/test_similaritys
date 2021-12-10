import pandas as pd
import numpy as np


def parse_data(path):
    global df, symbol, splited_data, block1, block2, block3, ans,full_target, full_target_df
    symbol = '*'
    df = pd.read_excel(path, dtype=str)
    df.loc[:, 'digits'] = 0
    splited_data = df["c_response"].str.split("/", n=3, expand=True)
    block1 = splited_data[0]
    block2 = splited_data[1]
    block3 = splited_data[2]
    ans = block1.map(str) + block2.map(str) + block3.map(str)
    full_target = df['target1'].map(str) + df['target2'].map(str) + df['target3'].map(str)
    full_target_df = full_target.str.split(pat="\s*", expand=True)  # .reset_index()
    full_target_df = full_target_df.iloc[:, 1:-1]


def add_grate(block):
    add = block.apply(lambda x: 4 if x == '*' else 0)
    return add


def add_exact(b_correct, block, target):
    for i in range(block.shape[0]):
        sum_similer = 0
        if block[i] == symbol:
            sum_similer = 4
        else:
            for j in range(len(block[i])):
                if block[i][j] == target[i][j]:
                    sum_similer += 1
        b_correct[i] = sum_similer


def add_rest(b_correct, block, target):
    for i in range(block.shape[0]):
        sum_similer = 0
        items = [item for item in block[i]]
        block_dict = {item: items.count(item) for item in items}
        t_block = [i for i in target[i]]
        for val in block_dict:
            if val == symbol:
                sum_similer = 0
            else:
                sum_similer += min(block_dict[val], t_block.count(val))
        b_correct.iloc[i] += sum_similer


def n_correct_digits_in_correct_quartet():
    b1_correct = add_grate(block=block1)
    b2_correct = add_grate(block=block2)
    b3_correct = add_grate(block=block3)
    add_rest(b1_correct, block=block1, target=df['target1'])
    add_rest(b2_correct, block=block2, target=df['target2'])
    add_rest(b3_correct, block=block3, target=df['target3'])
    df['n_correct_digits_in_quartet_1'] = b1_correct
    df['n_correct_digits_in_quartet_2'] = b2_correct
    df['n_correct_digits_in_quartet_3'] = b3_correct
    df['n_correct_digits_in_correct_quartet'] = df[['n_correct_digits_in_quartet_1'
        , 'n_correct_digits_in_quartet_2'
        , 'n_correct_digits_in_quartet_3']].astype(int).sum(1)

    # print(df['n_correct_digits_in_correct_quartet'])


def digits():
    correct = add_grate(block1)
    correct += add_grate(block2)
    correct += add_grate(block3)
    add_rest(correct, ans, full_target)
    df['digits'] = correct


def n_correct_digits_in_incorrect_quartet():
    df['n_correct_digits_in_incorrect_quartet'] = df['n_correct_digits_in_correct_quartet'].astype(int) * -1
    df['n_correct_digits_in_incorrect_quartet'] = df[['digits'
        , 'n_correct_digits_in_incorrect_quartet']].astype(int).sum(1)
    # print(df['n_correct_digits_in_incorrect_quartet'])


def n_correct_digits_oreder_absolute():
    count1 = np.arange(block1.shape[0])
    count2 = np.arange(block1.shape[0])
    count3 = np.arange(block1.shape[0])
    add_exact(b_correct=count1, block=block1, target=df['target1'])
    add_exact(b_correct=count2, block=block2, target=df['target2'])
    add_exact(b_correct=count3, block=block3, target=df['target3'])
    df['n_correct_digits_oreder_absolute'] = count1 + count2 + count3


if __name__ == '__main__':
    path = "C:\\Users\\elads\\Downloads\\__results same example 2.xlsx"
    parse_data(path)
    n_correct_digits_in_correct_quartet()
    digits()
    n_correct_digits_in_incorrect_quartet()
    n_correct_digits_oreder_absolute()
    path = "C:\\Users\\elads\\Downloads\\__results same example result.xlsx"
    df.to_excel(path)

