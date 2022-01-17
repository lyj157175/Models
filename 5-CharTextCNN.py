import torch
import torch.nn as nn 



class ChartextCNN(nn.Module):
    '''6层卷积，3层全连接层'''

    def __init__(self, config):
        super(ChartextCNN, self).__init__()
        self.in_features = [config.num_chars] + config.features[:-1]  # [70, 256, 256, 256, 256, 256]
        self.out_features = config.features    # [256, 256, 256, 256, 256, 256]
        self.kernel_sizes = config.kernel_sizes   # [7,7,3,3,3,3]
        self.dropout = config.dropout

        # conv1d(embeding, num_filters, filer_size)
        # out = Conv1d(x) out: b, num_filetrs, (n+2p-f)/s + 1
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(self.in_features[0], self.out_features[0], self.kernel_sizes[0], stride=1), 
            nn.BatchNorm1d(self.out_features[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(self.in_features[1], self.out_features[1], self.kernel_sizes[1], stride=1),
            nn.BatchNorm1d(self.out_features[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(self.in_features[2], self.out_features[2], self.kernel_sizes[2], stride=1),
            nn.BatchNorm1d(self.out_features[2]),
            nn.ReLU()
        )
        self.conv1d_4 = nn.Sequential(
            nn.Conv1d(self.in_features[3], self.out_features[3], self.kernel_sizes[3], stride=1),
            nn.BatchNorm1d(self.out_features[3]),
            nn.ReLU()
        )
        self.conv1d_5 = nn.Sequential(
            nn.Conv1d(self.in_features[4], self.out_features[4], self.kernel_sizes[4], stride=1),
            nn.BatchNorm1d(self.out_features[4]),
            nn.ReLU()
        )
        self.conv1d_6 = nn.Sequential(
            nn.Conv1d(self.in_features[5], self.out_features[5], self.kernel_sizes[5], stride=1),
            nn.BatchNorm1d(self.out_features[5]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.fc3 = nn.Linear(1024, config.num_labels)


    def forward(self, x):
        # x = torch.Tensor(x).long()   # b, num_chars, seq_len
        x = self.conv1d_1(x)   # b, out_features[0], （seq_len-f + 1）-f/s+1  = 64, 256, (1014-7+1)-3/3 + 1=1008-3/3+1=336
        x = self.conv1d_2(x)   # 64, 256, (336-7+1)-3/3+1=110
        x = self.conv1d_3(x)   # 64, 256, 110-3+1=108
        x = self.conv1d_4(x)   # 64, 256, 108-3+1=106
        x = self.conv1d_5(x)   # 64, 256, 106-3=1=104
        x = self.conv1d_6(x)   # 64, 256, (104-3+1)-3/3+1=34

        x = x.view(x.size(0), -1)   # 64, 256, 34 -> 64, 8704
        out = self.fc1(x)           # 64, 1024
        out = self.fc2(out)         # 64, 1024
        out = self.fc3(out)         # 64, 4
        return out 



class Config:
    def __init__(self):
        self.num_chars = 70
        self.features = [256, 256, 256, 256, 256, 256]
        self.kernel_sizes = [7, 7, 3, 3, 3, 3]
        self.dropout = 0.5
        self.num_labels = 4


if __name__ == '__main__':
    config = Config()
    model = ChartextCNN(config)
    # print(model)

    x = torch.zeros([64, 70, 1014])  # b, num_chars, seq_len
    out = model(x)
    print(out.shape) 

