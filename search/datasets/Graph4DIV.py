class GraphDataset(Dataset):
    def __init__(self,  graph_list):
        self.data = graph_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self,  idx):
        feat = self.data[idx][0]
        w = self.data[idx][1]
        rel_feat_tensor = torch.tensor(self.data[idx][2]).float()
        pos_mask = self.data[idx][3].bool()
        neg_mask = self.data[idx][4].bool()
        A = self.data[idx][5].float()
        degree_tensor = self.data[idx][6].float()
        feat.requires_grad = False
        rel_feat_tensor.requires_grad = False
        pos_mask.requires_grad = False
        neg_mask.requires_grad = False
        A.requires_grad = False
        degree_tensor.requires_grad = False
        return A, feat, rel_feat_tensor, degree_tensor, pos_mask, neg_mask, w


class Dataset(TensorDataset):
    def __init__(self, X_input_ids1, X_attention_mask1, X_token_type_ids1, X_input_ids2, X_attention_mask2, X_token_type_ids2, y_labels=None):
        super(Dataset, self).__init__()
        X_input_ids1 = torch.LongTensor(X_input_ids1)
        X_attention_mask1 = torch.LongTensor(X_attention_mask1)
        X_token_type_ids1 = torch.LongTensor(X_token_type_ids1)
        X_input_ids2 = torch.LongTensor(X_input_ids2)
        X_attention_mask2 = torch.LongTensor(X_attention_mask2)
        X_token_type_ids2 = torch.LongTensor(X_token_type_ids2)
        if y_labels is not None:
            y_labels = torch.FloatTensor(y_labels)
            self.tensors = [X_input_ids1, X_attention_mask1, X_token_type_ids1, X_input_ids2, X_attention_mask2, X_token_type_ids2, y_labels]
        else:
            self.tensors = [X_input_ids1, X_attention_mask1, X_token_type_ids1, X_input_ids2, X_attention_mask2, X_token_type_ids2]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])