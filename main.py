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
def add_rest(b_correct, block, target, func_code):
    t = target.copy()
    for i in range(block.shape[0]):
        if i == 17:
            print(" ")
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
    add_rest(b1_correct, block=block1, target=df['target1'], func_code=0)
    add_rest(b2_correct, block=block2, target=df['target2'], func_code=0)
    add_rest(b3_correct, block=block3, target=df['target3'], func_code=0)

    df['n_correct_digits_in_correct_quartet'] = (b1_correct + b2_correct + b3_correct)


def digits():
    correct = add_great(block1)
    correct += add_great(block2)
    correct += add_great(block3)
    add_rest(correct, ans, full_target, func_code=1)
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


# ////////////////////////////////////////////////
def count_good_blocks():  # count all cell without X
    summ = good_block(block=block1) + good_block(block=block2) + good_block(block=block3)
    df['sum_4_digits)in_quartet'] = summ


def good_block(block):  # 0 if has X in it , 1 else
    add = 1 - block.str.contains('X', regex=False).astype(int)
    return add


def block_dup(block, target):  # return a mapping of all dups and valid block
    add = block.apply(lambda x: 0 if len(set(x)) == 4 or len(set(x)) == 1 else 1)
    return extract_block_dup(add & good_block(block), block, target)


"""
:param 
        @bool_block : block tells us when there where duplicats in valid blocks
        @block : ans of student
        @target : target 
        
:return 

        bool block in witch duplicated chars are diffetent from ans and target
"""


def extract_block_dup(bool_block, block, target):
    for i in range(len(bool_block)):
        if '+' in block[i]:
            bool_block[i] = 0
            continue
        if i == 7:
            print()
        if bool_block[i] != 0:
            if len(set(target[i])) == 4:
                continue
            else:
                target_occ_dict = {x: target[i].count(x) for x in set(target[i])}
                ans_occ_dict = {x: block[i].count(x) for x in set(block[i])}
                for x in ans_occ_dict:
                    if ans_occ_dict[x] > 1:
                        if x not in target_occ_dict or target_occ_dict[x] < 2:
                            bool_block[i] = 1
                            continue
                        else:
                            bool_block[i] = 0

    return bool_block


def count_dup():  # caller func to gether number of dup cells in col
    df['sum_4_digits_in_quartet'] = block_dup(block=block1, target=df['target1']) + block_dup(block=block2, target=df[
        'target2']) + block_dup(block=block3, target=df['target3'])


def leeks():
    target1 = df['target1']
    target2 = df['target2']
    target3 = df['target3']
    target1_c = target2.str.cat(target3)
    target2_c = target1.str.cat(target3)
    target3_c = target1.str.cat(target2)

    df['n_digits_from_other_quartet'] = block_leek(block1, target1, target1_c) + block_leek(block2, target2,target2_c) + block_leek(block3, target3, target3_c)


def block_leek(block, target, target_c):
    # out = pd.DataFrame(0, index=np.arange(len(block)))
    # out =(block * 0).astype(int)
    out = block.apply(lambda x: 0)
    for i in range(len(block)):
        block_dict = {x: block[i].count(x) for x in set(block[i])}
        target_dict = {x: target[i].count(x) for x in set(target[i])}
        target_c_dict = {x: target_c[i].count(x) for x in set(target_c[i])}
        for x in block_dict:
            if '+' in block[i]:
                continue
            if x in target_dict:
                x_num_left = block_dict[x] - target_dict[x]
                if x_num_left <= 0:  # got enough of char x
                    continue
                if x in target_c_dict:  # remainder of whats left
                    out[i] = min(x_num_left, target_c_dict[x])
            elif x in target_c_dict:  # if i dont have it in the first place then get the min of its occurrences
                out[i] = min(block_dict[x], target_c_dict[x])
    return out


if __name__ == '__main__':
    path = path_find()
    parse_data(path)
    n_correct_digits_in_correct_quartet()
    digits()
    n_correct_digits_in_incorrect_quartet()
    n_correct_digits_oreder_absolute()
    count_good_blocks()
    count_dup()
    leeks()
    new_path = new_path(path)
    df.to_excel(new_path)
    print(
        "conversion of file :\t" + str(path.split('/')[-1]) + "\nnew file is named :\t" + str(new_path.split('/')[-1]))
    input("press Enter to end")
# auto-py-to-exe
