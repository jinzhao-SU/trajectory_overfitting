import torch
import torch.nn as nn
import torchvision


class PreTrain(nn.Module):
    def __init__(self):
        super(PreTrain, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stn_conv1 = nn.Conv1d(4, 64, 1)
        self.stn_conv2 = nn.Conv1d(64, 128, 1)
        self.stn_conv3 = nn.Conv1d(128, 1024, 1)

        self.stn_fc1 = nn.Linear(1024, 512)
        self.stn_fc2 = nn.Linear(512, 256)
        self.stn_fc3 = nn.Linear(256, 25)

        self.stn_bn1 = nn.BatchNorm1d(64)
        self.stn_bn2 = nn.BatchNorm1d(128)
        self.stn_bn3 = nn.BatchNorm1d(1024)
        self.stn_bn4 = nn.BatchNorm1d(512)
        self.stn_bn5 = nn.BatchNorm1d(256)

        # pNet embedding model declaration
        self.pNet_conv1 = torch.nn.Conv1d(4, 64, 1)
        self.pNet_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.pNet_conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.pNet_bn1 = nn.BatchNorm1d(64)
        self.pNet_bn2 = nn.BatchNorm1d(128)
        self.pNet_bn3 = nn.BatchNorm1d(1024)

        self.pNet_fc1 = nn.Linear(32*32, 4096)
        self.pNet_fc2 = nn.Linear(4096, 4096*4)
        self.pNet_fc3 = nn.Linear(4096*4, 100*100)

    def pNet_forward(self, x):
        trans = self.stn_forward(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = self.relu(self.pNet_bn1(self.pNet_conv1(x)))
        x = self.relu(self.pNet_bn2(self.pNet_conv2(x)))
        x = self.pNet_bn3(self.pNet_conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)

        x = self.pNet_fc1(x)
        x = self.pNet_fc2(x)
        x = self.pNet_fc3(x)


        x = x.view(-1,100,100)

        return x

    # STN to support the pNet
    def stn_forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.stn_bn1(self.stn_conv1(x)))
        x = self.relu(self.stn_bn2(self.stn_conv2(x)))
        x = self.relu(self.stn_bn3(self.stn_conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.stn_bn4(self.stn_fc1(x)))
        x = self.relu(self.stn_bn5(self.stn_fc2(x)))
        x = self.stn_fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 0,
        #                                            0, 1, 0, 0, 0,
        #                                            0, 0, 1, 0, 0,
        #                                            0, 0, 0, 1, 0,
        #                                            0, 0, 0, 0, 1]).astype(np.float32))).view(1, 25).repeat(batch_size,
        #                                                                                                    1)

        iden = torch.tensor([1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1], dtype=torch.float32).view(1, ).repeat(batch_size, 1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, 4, 4)
        return x

    def forward(self, x_image=None):
        x_image = x_image

        x = self._pNet_forward(x_image)
        return x

if __name__ == '__main__':
    test_tensor = torch.rand(2, 4, 1024)
    test = PreTrain()
    # print(test.stn_forward(test_tensor))
    print(test.pNet_forward(test_tensor).shape)
