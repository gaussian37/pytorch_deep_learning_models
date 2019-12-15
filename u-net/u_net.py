def ConvBlock(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def ConvTransBlock(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 3, stride = 2, padding=1, output_padding = 1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def Maxpool():
    pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    return pool

def ConvBlock2X(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        ConvBlock(in_dim, out_dim, act_fn),
        ConvBlock(out_dim, out_dim, act_fn),
    )
    return model

class UNet(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace = True)

        self.down_1 = ConvBlock2X(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = Maxpool()
        self.down_2 = ConvBlock2X(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = Maxpool()
        self.down_3 = ConvBlock2X(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = Maxpool()
        self.down_4 = ConvBlock2X(self.num_filter * 4, self.num_filter * 8, act_fn)
        self.pool_4 = Maxpool()
        
        self.bridge = ConvBlock2X(self.num_filter * 8, self.num_filter * 16, act_fn)

        self.trans_1 = ConvTransBlock(self.num_filter * 16,self.num_filter * 8, act_fn)
        self.up_1 = ConvBlock2X(self.num * 16, self.num_filter * 8, act_fn)
        self.trans_2 = ConvTransBlock(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.up_2 = ConvBlock2X(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.trans_3 = ConvTransBlock(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.up_3 = ConvBlock2X(self.num_filter * 2, self.num_filter, act_fn)
        self.trans_4 = ConvTransBlock(self.num_filter * 2, self.num_filter, act_fn)
        self.up_4 = ConvBlock2X(self.num_filter *2, self.num_filter, act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace = True),
        )

        def forward(self, input):
            down_1 = self.down_1(input) # concat w/ trans_4
            pool_1 = self.pool_1(down_1) 
            down_2 = self.down_2(pool_1) # concat w/ trans_3
            pool_2 = self.pool_2(down_2) 
            down_3 = self.down_3(pool_2) # concat w/ trans_2
            pool_3 = self.pool_3(down_3) 
            down_4 = self.down_4(pool_3) # concat w/ trans_1
            pool_4 = self.pool_4(down_4) 

            bridge = self.bridge(pool_4)

            trans_1 = self.trans_1(bridge)
            concat_1 = torch.cat([trans_1, down_4], dim = 1)
            up_1 = self.up_1(concat_1)
            trans_2 = self.trans_1(up_1)
            concat_2 = torch.cat([trans_2, down_3], dim = 1)
            up_2 = self.up_1(concat_2)
            trans_3 = self.trans_1(up_2)
            concat_3 = torch.cat([trans_3, down_2], dim = 1)
            up_3 = self.up_1(concat_3)
            trans_4 = self.trans_1(up_3)
            concat_4 = torch.cat([trans_4, down_1], dim = 1)
            up_4 = self.up_1(concat_4)
            out = self.out(up_4)
            return out