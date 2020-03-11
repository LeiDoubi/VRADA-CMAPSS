from cmapss_dataset import CMAPSSTrainingDataset, CMAPSSTestDataset
import os
from torch.autograd import Function

class NegativeDerivative(Function):



class DatasetWrapper:
    def __init__(self,
                 data_set_type,
                 data_set_index,
                 data_set_root='data/',
                 max_rul=125):
        """[summary]

        Args:
            data_set_type (string): 'train' or 'test'
            data_set_index ([int]): [1,2,3,4]
            data_set_root (str, optional):dir path containing data. Defaults to 'data/'.
            max_rul (int, optional): maximum remaining useful life. Defaults to 125.

        Raises:
            ValueError: [description]
        """
        # Select features according to https://doi.org/10.1016/j.ress.2017.11.02
        feature_idx = [2, 3, 6, 7, 8, 11, 12,
                       13, 15, 16, 17, 18, 19, 21, 24, 25]
        fileNames_test = ['test_FD001.txt',
                          'test_FD002.txt',
                          'test_FD003.txt',
                          'test_FD004.txt']
        fileNames_train = ['train_FD001.txt',
                           'train_FD002.txt',
                           'train_FD003.txt',
                           'train_FD004.txt']
        fileNames_tags_test = ['RUL_FD001.txt',
                               'RUL_FD002.txt',
                               'RUL_FD003.txt',
                               'RUL_FD004.txt']
        if data_set_type not in ['train', 'test']:
            raise ValueError('data_set_type must set to \'train\' or \'test\'')
        if data_set_type == 'train':
            data_path = os.path.join(
                data_set_root, fileNames_train[data_set_index-1])
            self.dataset = CMAPSSTrainingDataset(data_path,
                                                 max_rul,
                                                 feature_idx,
                                                 True
                                                 )
        else:
            data_path = os.path.join(
                data_set_root, fileNames_test[data_set_index-1]
            )
            tags_path = os.path.join(
                data_set_root, fileNames_tags_test[data_set_index-1]
            )
            self.dataset = CMAPSSTestDataset(data_path,
                                             tags_path,
                                             max_rul,
                                             feature_idx,
                                             True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    testdata = DatasetWrapper('test', 1, 'dataSet')
