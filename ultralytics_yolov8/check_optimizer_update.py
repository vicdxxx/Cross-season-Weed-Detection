import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

da_device = torch.device('cuda')

class netD_pixel(nn.Module):
    def __init__(self, context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.context = context
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
        normal_init(self.conv1, 0, 0.1)
        normal_init(self.conv2, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.sigmoid(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class netD(nn.Module):
    def __init__(self, context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(256, 128, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128, 32, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32, 2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 32)
        if self.context:
            feat = x
        x = self.fc(x)
        if self.context:
            return x, feat
        else:
            return x

class net_gcr(nn.Module):
    def __init__(self, out_channels=12):
        super(net_gcr, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.conv2 = conv3x3(512, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv3 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                # not a perfect approximation
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = self.conv3(x).squeeze(-1).squeeze(-1)
        return x

class net_gcr_simple(nn.Module):
    def __init__(self, in_channels=256, out_channels=12):
        super(net_gcr_simple, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x).squeeze(-1).squeeze(-1)
        return x

class L2Norm(nn.Module):
    def __init__(self, n_channels=256, scale=12):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


#net = netD_pixel()
#net = netD()
#net = net_gcr()
net = net_gcr_simple()
#net = L2Norm()

#shape_parameters_estimated = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
#shape_parameters_estimated=shape_parameters_estimated.to(da_device)
#stat_model = nn.Linear(1, 1)
#stat_model = stat_model.to(da_device)

learning_rate = 0.1
#optimizer = torch.optim.Adagrad([shape_parameters_estimated, stat_model.weight], lr=learning_rate)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
max_iter = 10
#criterion = nn.MSELoss()

for iteration in range(max_iter):
    #theta_estimated = shape_parameters_estimated
    #out = stat_model(theta_estimated)
    #loss = criterion(out, torch.rand_like(out))

    x = torch.randn(4, 256, 100, 100)
    #x = torch.randn(4, 512, 100, 100)
    domain_l = net(x)
    dloss_l = 0.5 * torch.mean(domain_l ** 2)
    loss = dloss_l
    #print('theta=', theta_estimated[0])
    #print('weight=', stat_model.weight)
    print(iteration)
    print(loss)
    print(net.conv1.weight[0, 0, 0, 0])
    #print(net.weight[0])
    #print(net.weight.requires_grad)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()