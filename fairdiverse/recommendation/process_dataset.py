import numpy as np
import pandas as pd
import os
import random
import copy
import yaml
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm,trange

def build_map(df,col_name):
    """
        Encodes a categorical column in a DataFrame by mapping unique values to integers
        and returns a decoding dictionary.

        :param df: The input DataFrame.
        :param col_name: The name of the column to be encoded.
        :return: dict: A dictionary mapping encoded integers back to original values.
    """
    key = df[col_name].unique().tolist()
    encode = dict(zip(key, range(len(key))))
    decode = dict(zip(range(len(key)),key))
    df[col_name] = df[col_name].apply(lambda x: encode[x])
    return decode

def build_item_map(df,col_name, history_name):
    """
        Encodes a categorical column in a DataFrame by mapping unique values with historical records to integers
        and returns a decoding dictionary.

        :param df: The input DataFrame.
        :param col_name: The name of the column to be encoded.
        :return: dict: A dictionary mapping encoded integers back to original values.
    """


    item_set = []
    for i in range(len(df)):
        item_set.append(df[col_name][i])
        for h_i in df[history_name][i]:
            item_set.append(h_i)

    key = list(set(item_set))
    encode = dict(zip(key, range(len(key))))
    decode = dict(zip(range(len(key)),key))
    df[col_name] = df[col_name].apply(lambda x: encode[x])
    df[history_name] = df[history_name].apply(lambda x: [encode[i] for i in x])
    return decode, len(key)

def Process(dataset, config=None):
    """
        Pre-process dataset, split training-validate-test set into ~/processed_data path

        :param dataset: str: The utilized dataset.
            Note that please download it into ~/dataset path

        :param config: The custom config files.
    """


    with open(os.path.join("recommendation","properties", "dataset.yaml"), "r") as file:
        process_config = yaml.safe_load(file)


    with open(os.path.join("recommendation","properties", "dataset", "{}.yaml".format(dataset)), "r") as file:
        dataset_config = yaml.safe_load(file)

    process_config.update(dataset_config)

    if config:
        process_config.update(config)

    print("process config:")
    print(process_config)


    if os.path.exists(os.path.join("recommendation", "processed_dataset", str(dataset))) and process_config['reprocess'] == False:
        return "do not need to process"

    print("start to process data...")
    inter_file = os.path.join("recommendation","dataset", dataset, dataset+".inter")
    provider_file = os.path.join("recommendation","dataset", dataset, dataset + ".item")


    frames = pd.read_csv(inter_file,delimiter='\t',dtype={process_config["item_id"]:str,process_config["user_id"]:str, process_config['timestamp']:float},
                         usecols=[process_config["user_id"],process_config["item_id"],process_config["label_id"], process_config['timestamp']])
    frames = frames.dropna()
    #provider_frames = pd.read_csv(provider_file, delimiter='\t')
    #print(provider_frames.columns)
    #exit(0)
    provider_frames = pd.read_csv(provider_file,delimiter='\t',
                                  usecols=[process_config["item_id"],process_config["group_id"], process_config["text_id"]],
                                  dtype={process_config["item_id"]:str,process_config["group_id"]:str, process_config["text_id"]:str})

    provider_frames = provider_frames.dropna()


    print(len(provider_frames))

    frames = copy.copy(frames.sample(frac=process_config['sample_size'], random_state=42, axis=0, replace=False).reset_index(drop=True))


    #print("start to pre-process data...")
    uid_field,iid_field,label_field, time_field, text_field = \
        [process_config["user_id"],process_config["item_id"],process_config["label_id"], process_config['timestamp'], process_config["text_id"]]
    id2text = dict(zip(provider_frames[iid_field], provider_frames[text_field]))


    provider_field = process_config["group_id"]

    frames.drop_duplicates(subset=[iid_field,uid_field],keep='first',inplace=True)
    #print(uid_field,iid_field,label_field,time_field)
    item_num, user_num = len(frames[iid_field].unique()),len(frames[uid_field].unique())
    group_num = len(provider_frames[provider_field].unique())
    print("origin:----------item number: %d user number:%d group_num:%d total interactions:%d"%(item_num,user_num,group_num, len(frames)))



    print("start to merge data.....")

    print("processing item val")
    frames[label_field] = frames[label_field].apply(lambda x: 1 if x>=process_config['label_threshold'] else 0)

    frames.rename(columns={label_field:"label:float"},inplace=True)
    label_field = "label:float"
    frames = frames.merge(provider_frames,on=iid_field,how='inner')
    #print(len(frames))
    #exit(0)

    for i in range(3):
        itemLen = frames.groupby(iid_field).size()  # groupby itemID and get size of each item
        remain_items =  (itemLen[itemLen >= process_config['item_val']].index).values
        frames = frames[frames[iid_field].isin(remain_items)].reset_index(drop=True)
        #print(len(frames))
        #print(itemLen)
        userLen = frames.groupby(uid_field).size()
        remain_users =  (userLen[userLen >= process_config['user_val']].index).values
        frames = frames[frames[uid_field].isin(remain_users)].reset_index(drop=True)
        #print(len(frames))
        pidLen = frames.groupby(provider_field)[iid_field].unique().apply(lambda x: len(x))
        remain_providers =  (pidLen[pidLen >= process_config['group_val']].index).values
        frames = frames[frames[provider_field].isin(remain_providers)].reset_index(drop=True)

        #print(len(frames))
        #exit(0)

    frames = frames.sort_values(by=[uid_field, time_field]).reset_index(drop=True)

    def get_previous_items(sub_df):
        clicked_items = []
        previous_items = []

        # go through
        for _, row in sub_df.iterrows():
            #
            previous_items.append(clicked_items[-process_config['history_length']:])  # last k clicked items
            #
            if row[label_field] == 1:
                clicked_items.append(row[iid_field])


        sub_df['history_behaviors'] = previous_items
        return sub_df

    frames = frames.groupby(uid_field, group_keys=False).apply(get_previous_items)
    #print(frames)

    #
    frames = frames[
        frames['history_behaviors'].apply(lambda x: len(x) == process_config['history_length'])].reset_index(drop=True)
    history_field = 'history_behaviors'
    process_config['history_field'] = 'history_behaviors'

    user_num, group_num = len(frames[uid_field].unique()), len(frames[provider_field].unique())





    min_item_size = 99999
    max_item_size = -1

    #build_map(frames,iid_field)
    iid_decode, item_num = build_item_map(frames, iid_field, history_field)
    _ = build_map(frames,uid_field)


    pid2iid = {}
    iid2pid = {}
    for i, (iid, pid) in enumerate(zip(frames[iid_field].values, frames[provider_field].values)):
        if pid not in pid2iid:
            pid2iid[pid] = []
        pid2iid[pid].append(iid)
        iid2pid[iid] = pid
    for key in pid2iid.keys():
        pid2iid[key] = list(set(pid2iid[key]))


    ###Since some group contains few items, we need to aggregate the some groups into one group
    ##SO WE NEED TO remap the provider id
    #index = 1
    remap_dict = {}
    for pid in frames[provider_field].unique():
        items = pid2iid[pid]
        if len(items) < process_config['group_aggregation_threshold']:
            for i in items:
                remap_dict[i] = -1
    iid2pid.update(remap_dict)
    #print(iid2pid)

    frames['remap_pid'] = frames[iid_field].map(iid2pid)
    provider_field = 'remap_pid'
    build_map(frames, provider_field)

    pid2iid = {}
    iid2pid = {}
    for i, (iid, pid) in enumerate(zip(frames[iid_field].values, frames[provider_field].values)):
        iid = str(iid)
        pid = str(pid)
        if pid not in pid2iid:
            pid2iid[pid] = []
        pid2iid[pid].append(iid)
        iid2pid[iid] = pid
    for key in pid2iid.keys():
        pid2iid[key] = str(list(set(pid2iid[key])))

    # item_num, user_num, group_num = len(frames[iid_field].unique()), len(frames[uid_field].unique()), len(
    #     frames[provider_field].unique())
    # print("after process.... item number: %d user number:%d group number:%d total interactions:%d" % (
    # item_num, user_num, group_num, len(frames)))
    group_num = len(frames[provider_field].unique())

    process_config['user_num'] = user_num
    process_config['item_num'] = item_num
    process_config['group_num'] = group_num
    process_config['label_id'] = label_field

    print("item number: %d user number:%d group number:%d total interactions:%d" % (
        item_num, user_num, group_num, len(frames)))



    print("start to sample and split data....")
    frames = frames.sample(frac=1, random_state=42).reset_index(drop=True)
    #frames = frames.reset_index(drop=True)


    #print(frames.head())
    train_items = frames[iid_field].unique()
    train_users = frames[uid_field].unique()
    print("checking.....")


    if np.sum(train_users>=len(train_users)) > 0:
        print("ERROR in user id:",np.sum(train_users>=len(train_users)))
        exit(0)

    print("min user interactions:",np.min(frames.groupby(uid_field).size()))
    print("min item interactions:",np.min(frames.groupby(iid_field).size()))



    final_frame = frames[[uid_field, iid_field, history_field, label_field, provider_field, time_field]]
    final_frame.rename(columns={
        'remap_pid': process_config['group_id']
    }, inplace=True)


    final_frame.sort_values(by=time_field, inplace=True)

    train_size = int(len(final_frame) * (1 - process_config['valid_ratio'] - process_config['test_ratio']))
    val_size = int(len(final_frame) * process_config['valid_ratio'])

    train = final_frame.iloc[:train_size]

    val = final_frame.iloc[train_size:train_size + val_size]
    test = final_frame.iloc[train_size + val_size:]

    print(test.head())
    # train, temp = train_test_split(final_frame, test_size=(1 - process_config['valid_ratio'] - process_config['test_ratio']), random_state=42)
    # val, test = train_test_split(temp, test_size=(process_config['test_ratio'] / (process_config['valid_ratio'] + process_config['test_ratio'])), random_state=42)

    #split train-val-test accoring to the time


    # This is for extracting the positive items for every user to make it easy to conduct negative samples for each users
    user2pos_itemset = {}
    for i in range(len(final_frame)):
        uid = final_frame[uid_field][i]
        iid = final_frame[iid_field][i]
        label = final_frame[label_field][i]
        if uid not in user2pos_itemset.keys():
            user2pos_itemset[uid] = []
        if label == 1:
            user2pos_itemset[uid].append(iid)


    dir = os.path.join("recommendation","processed_dataset",dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)



    #columns = [process_config['user_id'], process_config['item_id'], process_config['label_id'], process_config['group_id']]


    test.to_csv(os.path.join(dir,dataset+".test.CTR"), index=False,sep='\t')
    val.to_csv(os.path.join(dir,dataset+".valid.CTR"), index=False,sep='\t')
    train.to_csv(os.path.join(dir,dataset+".train"), index=False,sep='\t')

    #iid2pid
    def construct_ranking_data(df):

        sort_df = df.sort_values(by=[uid_field, time_field]).reset_index(drop=True)
        #construct_dict = {"user_id":[], "items":[], "pos_length":[] }

        construct_dict = {}
        user_behaviors = {}

        for i in trange(len(sort_df)):

            user, item, label = sort_df[uid_field].values[i], sort_df[iid_field].values[i], sort_df[label_field].values[i]
            history = sort_df[history_field].values[i]
            ##find the behavior sequences in the training dataset
            if label == 1:
                if user not in construct_dict.keys():
                    construct_dict[user] = [item]
                    # filtered_df = train[(train[uid_field] == user) & (train[label_field] == 1)]
                    # print(filtered_df.head())
                    # filtered_df['time_diff'] = abs(filtered_df[time_field] - time)  # find the most close behavior
                    # closest_row = filtered_df.loc[filtered_df['time_diff'].idxmin()]
                    # behavior = closest_row["history_behaviors"][-(process_config['history_length']-1):]
                    # behavior.append(closest_row[iid_field])
                    user_behaviors[user] = history

                else:
                    construct_dict[user].append(item)

        df_dict = {"user_id":[], "history_behaviors":[], "items":[], "pos_length":[] }
        max_len = 0
        for u in tqdm(construct_dict.keys()):
            pos_length = len(construct_dict[u])
            itemset = construct_dict[u]
            if pos_length > max_len:
                max_len = pos_length

            for r in range(process_config['sample_num']-pos_length):
                neg_id = random.randint(0, item_num-1)
                while neg_id in user2pos_itemset[u]:
                    neg_id = random.randint(0, item_num - 1)
                itemset.append(neg_id)

            df_dict["user_id"].append(u)
            df_dict["history_behaviors"].append(user_behaviors[u])
            df_dict["items"].append(itemset)
            df_dict["pos_length"].append(pos_length)

        df_reconstruct = pd.DataFrame(df_dict)
        print(f"max_len:{max_len}")
        return df_reconstruct

    print("start to construct ranking and retrieval test dataset...")
    val_ranking = construct_ranking_data(val)
    test_ranking = construct_ranking_data(test)


    val_ranking.to_csv(os.path.join(dir,dataset+".valid.ranking"), index=False,sep='\t')
    test_ranking.to_csv(os.path.join(dir,dataset+".test.ranking"), index=False,sep='\t')

    id2text_update = {}
    for i in range(item_num):
        ori_i = iid_decode[i]
        text = id2text[ori_i]
        id2text_update[i] = text

    with open(os.path.join(dir, "process_config.yaml"), "w") as file:
        yaml.dump(process_config, file)

    #print(iid2pid)
    with open(os.path.join(dir, "iid2pid.json"), "w") as file:
        json.dump(iid2pid, file)

    with open(os.path.join(dir, "iid2text.json"), "w") as file:
        json.dump(id2text_update, file)


    return f"process complete! The file and config are stored in {dir}"


