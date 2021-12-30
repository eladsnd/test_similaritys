import pandas as pd
import numpy as np

from tkinter import Tk
from tkinter.filedialog import askopenfilename


def replace_substring(string, min_loc, max_loc, what_to):
    return string[0:min_loc] + what_to + string[max_loc + 1:]


def path_find():
    Tk().withdraw()
    filename = askopenfilename()
    return filename


def parse_data(path):
    global df, symbol, splited_data, block1, block2, block3, ans, full_target, full_target_df
    symbol = ['+', '\'+']
    df = pd.read_excel(path, dtype=str)
    df.columns = df.columns.str.replace(' ', '')
    df.loc[:, 'digits'] = 0
    splited_data = df["c_response"].str.split("/", n=3, expand=True)
    block1 = splited_data[0]
    block2 = splited_data[1]
    block3 = splited_data[2]
    ans = block1.map(str) + block2.map(str) + block3.map(str)
    full_target = df['target1'].map(str) + df['target2'].map(str) + df['target3'].map(str)
    full_target_df = full_target.str.split(pat="\s*", expand=True)  # .reset_index()
    full_target_df = full_target_df.iloc[:, 1:-1]


def add_great(block):
    add = block.apply(lambda x: 4 if x in symbol else 0)
    return add


def add_exact(b_correct, block, target):
    for i in range(block.shape[0]):
        sum_similer = 0
        if block[i] in symbol:
            sum_similer = 4
        else:
            for j in range(len(block[i])):
                if block[i][j] == target[i][j]:
                    sum_similer += 1
        b_correct[i] = sum_similer


# func_code 1 -> digits
# else -> rest
def add_rest(b_correct, block, target , func_code):
    t = target.copy()
    for i in range(block.shape[0]):
        if i == 17:
            print (" ")
        if func_code == 1:
            if len(block1[i]) < 4:
                t[i] = replace_substring(string=t[i], min_loc=0, max_loc=3, what_to='++++')
            if len(block2[i]) < 4:
                t[i] = replace_substring(string=t[i], min_loc=4, max_loc=7, what_to='++++')
            if len(block3[i]) < 4:
                t[i] = replace_substring(string=t[i], min_loc=8, max_loc=11, what_to='++++')

        sum_similer = 0
        items = [item for item in block[i]]
        block_dict = {item: items.count(item) for item in items}
        t_block = [i for i in t[i]]
        for val in block_dict:
            if val not in symbol:
                sum_similer += min(block_dict[val], t_block.count(val))
            if val in symbol:
                sum_similer += 0
        b_correct.iloc[i] += sum_similer


def n_correct_digits_in_correct_quartet():
    b1_correct = add_great(block=block1)
    b2_correct = add_great(block=block2)
    b3_correct = add_great(block=block3)
    add_rest(b1_correct, block=block1, target=df['target1'],func_code=0)
    add_rest(b2_correct, block=block2, target=df['target2'],func_code=0)
    add_rest(b3_correct, block=block3, target=df['target3'],func_code=0)
    # df['n_correct_digits_in_quartet_1'] = b1_correct
    # df['n_correct_digits_in_quartet_2'] = b2_correct
    # df['n_correct_digits_in_quartet_3'] = b3_correct
    # df['n_correct_digits_in_correct_quartet'] = df[['n_correct_digits_in_quartet_1'
    #     , 'n_correct_digits_in_quartet_2'
    #     , 'n_correct_digits_in_quartet_3']].astype(int).sum(1)
    df['n_correct_digits_in_correct_quartet'] = (b1_correct + b2_correct + b3_correct)

    # print(df['n_correct_digits_in_correct_quartet'])


def digits():
    correct = add_great(block1)
    correct += add_great(block2)
    correct += add_great(block3)
    add_rest(correct, ans, full_target,func_code=1)
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


def new_path(path):
    path = path.split('/')
    end = "/New_" + path[-1]
    return "/".join(path[:-1]) + end


if __name__ == '__main__':
    path = path_find()
    parse_data(path)

    n_correct_digits_in_correct_quartet()
    digits()
    n_correct_digits_in_incorrect_quartet()
    n_correct_digits_oreder_absolute()
    new_path = new_path(path)
    df.to_excel(new_path)
    print(
        "conversion of file :\t" + str(path.split('/')[-1]) + "\nnew file is named :\t" + str(new_path.split('/')[-1]))
    input("press Enter to end")
