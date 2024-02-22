import torch.nn as nn
import torch
from function import normal
from function import calc_mean_std
import scipy.stats as stats
from torchvision.utils import save_image

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class MCCNet(nn.Module):
    def __init__(self, in_dim):
        super(MCCNet, self).__init__()
        self.f = nn.Conv2d(in_dim , int(in_dim ), (1,1))
        self.g = nn.Conv2d(in_dim , int(in_dim ) , (1,1))
        self.h = nn.Conv2d(in_dim, int(in_dim), (1,1))
        #self.softmax  = nn.Softmax(dim=-1)    #16
        self.softmax  = nn.Softmax(dim=-2)    #17
        self.out_conv = nn.Conv2d(int(in_dim ), in_dim, (1, 1))
        self.fc = nn.Linear(in_dim, in_dim)
        #self.wNet = WNet()
    def forward(self,content_feat,style_feat):
        B,C,H,W = content_feat.size()

        F_Fc_norm  = self.f(normal(content_feat))
        
        #F_Fc_norm = torch.mul(F_Fc_norm, content_a.view(B,-1,H*W).permute(0,2,1))

        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(normal(style_feat)).view(-1,1,H*W)
        #print(G_Fs)
        #G_Fs_sum = torch.abs(G_Fs_norm.view(B,C,H*W)).sum(-1)
        G_Fs_sum = G_Fs_norm.view(B,C,H*W).sum(-1)
        #print(G_Fs_sum.size())
        #print(G_Fs_norm)
        FC_S = torch.bmm(G_Fs_norm,G_Fs_norm.permute(0,2,1)).view(B,C) /G_Fs_sum  #14
        #FC_S = torch.bmm(self.softmax(G_Fs_norm),G_Fs_norm.permute(0,2,1)).view(B,C)  #16
        #FC_S = torch.bmm(G_Fs, self.softmax(G_Fs_norm.permute(0,2,1))).view(B,C)   #17
        #FC_S = torch.bmm(G_Fs, G_Fs_norm.permute(0,2,1)).view(B,C)/G_Fs_sum   #18
        FC_S = self.fc(FC_S).view(B,C,1,1)
        #print(G_Fs_norm.size(),style_a.size())
        #G_Fs_norm = torch.mul(G_Fs_norm,style_a.view(B,-1,H*W) )
        #print(F_Fc_norm.size(),G_Fs_norm.size(),)
        
        out = F_Fc_norm*FC_S
        B,C,H,W = content_feat.size()
        out = out.contiguous().view(B,-1,H,W)
        out = self.out_conv(out)
        out = content_feat + out

        return out
 

class MCC_Module(nn.Module):
    def __init__(self, in_dim):
        super(MCC_Module, self).__init__()
        self.MCCN=MCCNet(in_dim)

    def forward(self, content_feats, style_feats):
        content_feat_4 = content_feats[-2]
        style_feat_4 = style_feats[-2]
        Fcsc = self.MCCN(content_feat_4, style_feat_4)
       
        return Fcsc

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.mcc_module = MCC_Module(512)
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      #assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, content, style):
        s = torch.empty(1)
        t  = torch.empty(content.size())

        std = torch.nn.init.uniform_(s, a=0.01, b=0.02)
        noise = torch.nn.init.normal(t, mean=0, std=std[0]).cuda()
        content_noise = content + noise

        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        content_feats_N = self.encode_with_intermediate(content_noise)

        Ics = self.decoder(self.mcc_module(content_feats, style_feats))
        Ics_feats = self.encode_with_intermediate(Ics)
        # Content loss
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        # total variation loss
        y = Ics
        tv_loss = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

        Ics_N = self.decoder(self.mcc_module(content_feats_N, style_feats))
        loss_noise = self.calc_content_loss(Ics_N,Ics)

        #Identity losses lambda 1
        Icc = self.decoder(self.mcc_module(content_feats, content_feats))
        Iss = self.decoder(self.mcc_module(style_feats, style_feats)) 

        loss_lambda1 = self.calc_content_loss(Icc,content)+self.calc_content_loss(Iss,style)
        
        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])
        return loss_noise, loss_c, loss_s, loss_lambda1, loss_lambda2,tv_loss, Ics
        #return loss_c, loss_s,loss_lambda1, loss_lambda2, tv_loss

