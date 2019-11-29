def conv_block_1(in_dim, out_dim, act_fn, stride = 1):
    """  bottleneck 구조를 만들기 위한 1x1 convolution """
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 1, stride = stride),
        act_fn
    )
    return model

def conv_block_3(in_dim, out_dim, act_fn):
    """  bottleneck 구조를 만들기 위한 3x3 convolution """
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = 1, padding = 1),
        act_fn
    )

class BottleNeck(nn.Module):
    """
    Residual Network의 bottleneck 구조
    - down 파라미터는 down 블록을 통과하였을 때, feature map의 크기가 줄어드는 지의 여부의 불리언 값 입니다.
        - True인 경우 stride = 2가 되어 크기가 반으로 줄어듭니다.
    - 경우에 따라서 채널의 갯수가 달라 더해지지 않는 경우가 있는데, 이럴 때는 차원을 맞춰 주는 1x1 conv를 추가하여
      입력의 채널을 출력의 채널과 같게 만들어 줍니다.
     """
    def __init__(self, in_dim, mid_dim, out_dim, act_fn, down=False):
        super(BottleNeck, self).__init__()
        self.act_fn = act_fn
        self.down = down

        if self.down:
            self.layer == nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn, 2),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn)
            )            
            self.downsample = nn.Conv2d(in_dim, out_dim, 1, 2)
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn)
            )
        # shape을 맞추기 위한 1x1 conv
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size = 1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out


class ResNet(nn.Module):
    """ ResNet 50 layer """
    def __init__(self, base_dim, num_classes = 2):
        super(ResNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size, stride, padding)
            nn.MaxPool2d(3, 2, 1),
        )    
        self.layer_2 = nn.Sequential(
            # BottleNeck(in_dim, mid_dim, out_dim, act_fn, down)
            BottleNeck(base_dim, base_dim, base_dim * 4, self.act_fn),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.act_fn),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.act_fn, down = True),
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn, down = True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn, act_fn = True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.act_fn),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.act_fn),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.act_fn),
        )
        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc_layer = nn.Linear(base_dim * 32, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out