import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)

        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self, x):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = self.relu2(out)
        out = self.reflecPad7(output['r11'])

        # out = self.reflecPad3(output['r11'])
        out = self.conv3(out)
        out = self.relu3(out)

        out = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21'])

        # out = self.reflecPad5(output['r21'])
        out = self.conv5(out)
        out = self.relu5(out)

        out = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        output['r31'] = self.relu6(out)

        out = self.reflecPad7(output['r31'])
        out = self.conv7(out)
        out = self.relu7(out)

        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)

        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)

        out = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        output['r41'] = self.relu10(out)
        out = self.reflecPad11(output['r41'])
        out = self.conv11(out)
        out = self.relu11(out)

        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)

        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)

        out = self.maxPool4(out)

        out = self.reflecPad14(out)
        out = self.conv14(out)
        output['r51'] = self.relu14(out)
        return output


class Decoder(nn.Module):
    def __init__(self, layer):
        super(Decoder, self).__init__()
        # decoder
        # r51
        flag = False
        if layer == 'r51' or flag:
            self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu15 = nn.ReLU(inplace=True)

            self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
            # 28 x 28

            self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu16 = nn.ReLU(inplace=True)
            # 28 x 28

            self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu17 = nn.ReLU(inplace=True)
            # 28 x 28

            self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu18 = nn.ReLU(inplace=True)
            # 28 x 28
            flag = True

        # r41
        if layer == 'r41' or flag:
            self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu19 = nn.ReLU(inplace=True)
            # 28 x 28

            self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
            # 56 x 56

            self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu20 = nn.ReLU(inplace=True)
            # 56 x 56

            self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu21 = nn.ReLU(inplace=True)

            self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu22 = nn.ReLU(inplace=True)
            flag = True

        # r31
        if layer == 'r31' or flag:
            self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu23 = nn.ReLU(inplace=True)

            self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
            # 112 X 112

            self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
            self.relu24 = nn.ReLU(inplace=True)
            flag = True

        # r21
        if layer == 'r21' or flag:
            self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu25 = nn.ReLU(inplace=True)

            self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

            self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
            self.relu26 = nn.ReLU(inplace=True)
            flag = True

        # r11
        if layer == 'r11' or flag:
            self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)
            flag = True

    def forward(self, x, layer):
        flag = False
        # r51
        if layer == 'r51' or flag:
            if not flag:
                out = x
            out = self.reflecPad15(out)
            out = self.conv15(out)
            out = self.relu15(out)
            out = self.unpool(out)
            out = self.reflecPad16(out)
            out = self.conv16(out)
            out = self.relu16(out)
            out = self.reflecPad17(out)
            out = self.conv17(out)
            out = self.relu17(out)
            out = self.reflecPad18(out)
            out = self.conv18(out)
            out = self.relu18(out)
            flag = True
        # r41
        if layer == 'r41' or flag:
            if not flag:
                out = x
            out = self.reflecPad19(out)
            out = self.conv19(out)
            out = self.relu19(out)
            out = self.unpool2(out)
            out = self.reflecPad20(out)
            out = self.conv20(out)
            out = self.relu20(out)
            out = self.reflecPad21(out)
            out = self.conv21(out)
            out = self.relu21(out)
            out = self.reflecPad22(out)
            out = self.conv22(out)
            out = self.relu22(out)
            flag = True
        # r31
        if layer == 'r31' or flag:
            if not flag:
                out = x
            out = self.reflecPad23(out)
            out = self.conv23(out)
            out = self.relu23(out)
            out = self.unpool3(out)
            out = self.reflecPad24(out)
            out = self.conv24(out)
            out = self.relu24(out)
            flag = True
        # r21
        if layer == 'r21' or flag:
            if not flag:
                out = x
            out = self.reflecPad25(out)
            out = self.conv25(out)
            out = self.relu25(out)
            out = self.unpool4(out)
            out = self.reflecPad26(out)
            out = self.conv26(out)
            out = self.relu26(out)
            flag = True
        # r11
        if layer == 'r11' or flag:
            if not flag:
                out = x
            out = self.reflecPad27(out)
            out = self.conv27(out)
            flag = True
        return out


class DecoderWithSkipConcatConnection(nn.Module):
    def __init__(self, layer_list):
        super(DecoderWithSkipConcatConnection, self).__init__()

        # decoder
        # r51
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        # r41
        if 'r41' in layer_list:
            self.reflecPad_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_1 = nn.Conv2d(1024, 512, 3, 1, 0)
            self.relu_1 = nn.ReLU(inplace=True)
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)

        # r31
        if 'r31' in layer_list:
            self.reflecPad_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_2 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu_2 = nn.ReLU(inplace=True)
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu24 = nn.ReLU(inplace=True)

        # r21
        if 'r21' in layer_list:
            self.reflecPad_3 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_3 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu_3 = nn.ReLU(inplace=True)
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu26 = nn.ReLU(inplace=True)

        # r11
        if 'r11' in layer_list:
            self.reflecPad_4 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_4 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu_4 = nn.ReLU(inplace=True)
        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    def freeze_params(self, layer_list):
        for p in self.parameters():
            p.requires_grad = False
        if 'r41' in layer_list:
            self.conv_1.weight.requires_grad = True
            self.conv_1.bias.requires_grad = True
        if 'r31' in layer_list:
            self.conv_2.weight.requires_grad = True
            self.conv_2.bias.requires_grad = True
        if 'r21' in layer_list:
            self.conv_3.weight.requires_grad = True
            self.conv_3.bias.requires_grad = True
        if 'r11' in layer_list:
            self.conv_4.weight.requires_grad = True
            self.conv_4.bias.requires_grad = True

    def forward(self, x, layer_list, concat_weight=1.0):
        # r51
        out = self.reflecPad15(x['r51'])
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        # r41
        if 'r41' in layer_list:
            out = torch.cat((concat_weight * x['r41'], out), dim=1)  # start
            out = self.reflecPad_1(out)
            out = self.conv_1(out)
            out = self.relu_1(out)  # end

        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        # r31
        if 'r31' in layer_list:
            out = torch.cat((concat_weight * x['r31'], out), dim=1)  # start
            out = self.reflecPad_2(out)
            out = self.conv_2(out)
            out = self.relu_2(out)  # end

        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        # r21
        if 'r21' in layer_list:
            out = torch.cat((concat_weight * x['r21'], out), dim=1)  # start
            out = self.reflecPad_3(out)
            out = self.conv_3(out)
            out = self.relu_3(out)  # end
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        # r11
        if 'r11' in layer_list:
            out = torch.cat((concat_weight * x['r11'], out), dim=1)  # start
            out = self.reflecPad_4(out)
            out = self.conv_4(out)
            out = self.relu_4(out)  # end
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out


class DecoderWithSkipConcatConnectionAdaIN(nn.Module):
    def __init__(self, layer_list):
        super(DecoderWithSkipConcatConnectionAdaIN, self).__init__()

        # decoder
        # r51
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        # r41
        if 'r41' in layer_list:
            self.reflecPad_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_1 = nn.Conv2d(1024, 512, 3, 1, 0)
            self.relu_1 = nn.ReLU(inplace=True)
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)

        # r31
        if 'r31' in layer_list:
            self.reflecPad_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_2 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu_2 = nn.ReLU(inplace=True)
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu24 = nn.ReLU(inplace=True)

        # r21
        if 'r21' in layer_list:
            self.reflecPad_3 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_3 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu_3 = nn.ReLU(inplace=True)
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu26 = nn.ReLU(inplace=True)

        # r11
        if 'r11' in layer_list:
            self.reflecPad_4 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv_4 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu_4 = nn.ReLU(inplace=True)
        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    def freeze_params(self, layer_list):
        for p in self.parameters():
            p.requires_grad = False
        if 'r41' in layer_list:
            self.conv_1.weight.requires_grad = True
            self.conv_1.bias.requires_grad = True
        if 'r31' in layer_list:
            self.conv_2.weight.requires_grad = True
            self.conv_2.bias.requires_grad = True
        if 'r21' in layer_list:
            self.conv_3.weight.requires_grad = True
            self.conv_3.bias.requires_grad = True
        if 'r11' in layer_list:
            self.conv_4.weight.requires_grad = True
            self.conv_4.bias.requires_grad = True

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, x, layer_list, concat_weight=1.0):
        # r51
        out = self.reflecPad15(x['r51'])
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        # r41
        if 'r41' in layer_list:
            target = self.adaptive_instance_normalization(x['r41'], out)
            out = torch.cat((concat_weight * target, out), dim=1)  # start
            out = self.reflecPad_1(out)
            out = self.conv_1(out)
            out = self.relu_1(out)  # end

        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        # r31
        if 'r31' in layer_list:
            target = self.adaptive_instance_normalization(x['r31'], out)
            out = torch.cat((concat_weight * target, out), dim=1)  # start
            out = self.reflecPad_2(out)
            out = self.conv_2(out)
            out = self.relu_2(out)  # end

        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        # r21
        if 'r21' in layer_list:
            target = self.adaptive_instance_normalization(x['r21'], out)
            out = torch.cat((concat_weight * target, out), dim=1)  # start
            out = self.reflecPad_3(out)
            out = self.conv_3(out)
            out = self.relu_3(out)  # end
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        # r11
        if 'r11' in layer_list:
            target = self.adaptive_instance_normalization(x['r11'], out)
            out = torch.cat((concat_weight * target, out), dim=1)  # start
            out = self.reflecPad_4(out)
            out = self.conv_4(out)
            out = self.relu_4(out)  # end
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out


class DecoderWithSkipSumConnection(nn.Module):
    def __init__(self):
        super(DecoderWithSkipSumConnection, self).__init__()

        # decoder
        # r51
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        # r41
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)

        # r31
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu24 = nn.ReLU(inplace=True)

        # r21
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu26 = nn.ReLU(inplace=True)

        # r11
        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, layer_list):
        # r51
        out = self.reflecPad15(x['r51'])
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        # r41
        if 'r41' in layer_list:
            out = out + x['r41']
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        # r31
        if 'r31' in layer_list:
            out = out + x['r31']
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        # r21
        if 'r21' in layer_list:
            out = out + x['r21']
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        # r11
        if 'r11' in layer_list:
            out = out + x['r11']
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out


class DecoderAWSC2(nn.Module):
    def __init__(self, layer_list, connect_weight=0.5):
        super(DecoderAWSC2, self).__init__()
        self.start_layer = layer_list[0]
        self.layer_list = layer_list
        self.connect_weight = connect_weight

        # decoder
        if '5' <= self.start_layer[1]:
            # r51
            self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu15 = nn.ReLU(inplace=True)

            self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
            # 28 x 28

            self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu16 = nn.ReLU(inplace=True)
            # 28 x 28

            self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu17 = nn.ReLU(inplace=True)
            # 28 x 28

            self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu18 = nn.ReLU(inplace=True)
            # 28 x 28

        if '4' <= self.start_layer[1]:
            # r41
            self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu19 = nn.ReLU(inplace=True)
            # 28 x 28

            self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
            # 56 x 56

            self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu20 = nn.ReLU(inplace=True)
            # 56 x 56

            self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu21 = nn.ReLU(inplace=True)

            self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu22 = nn.ReLU(inplace=True)

        if '3' <= self.start_layer[1]:
            # r31
            self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu23 = nn.ReLU(inplace=True)

            self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
            # 112 X 112

            self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
            self.relu24 = nn.ReLU(inplace=True)

        if '2' <= self.start_layer[1]:
            # r21
            self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu25 = nn.ReLU(inplace=True)

            self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

            self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
            self.relu26 = nn.ReLU(inplace=True)

        if '1' <= self.start_layer[1]:
            # r11
            self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, x):
        # r51
        if 'r51' == self.start_layer:
            out = self.reflecPad15(x['r51'])
            out = self.conv15(out)
            out = self.relu15(out)
            out = self.unpool(out)
            out = self.reflecPad16(out)
            out = self.conv16(out)
            out = self.relu16(out)
            out = self.reflecPad17(out)
            out = self.conv17(out)
            out = self.relu17(out)
            out = self.reflecPad18(out)
            out = self.conv18(out)
            out = self.relu18(out)
        # r41
        if '4' <= self.start_layer[1]:
            if 'r41' == self.start_layer:
                out = x['r41']
            elif 'r41' in self.layer_list:
                target = self.adaptive_instance_normalization(x['r41'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad19(out)
            out = self.conv19(out)
            out = self.relu19(out)
            out = self.unpool2(out)
            out = self.reflecPad20(out)
            out = self.conv20(out)
            out = self.relu20(out)
            out = self.reflecPad21(out)
            out = self.conv21(out)
            out = self.relu21(out)
            out = self.reflecPad22(out)
            out = self.conv22(out)
            out = self.relu22(out)
        # r31
        if '3' <= self.start_layer[1]:
            if 'r31' == self.start_layer:
                out = x['r31']
            elif 'r31' in self.layer_list:
                target = self.adaptive_instance_normalization(x['r31'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad23(out)
            out = self.conv23(out)
            out = self.relu23(out)
            out = self.unpool3(out)
            out = self.reflecPad24(out)
            out = self.conv24(out)
            out = self.relu24(out)
        # r21
        if '2' <= self.start_layer[1]:
            if 'r21' == self.start_layer:
                out = x['r21']
            elif 'r21' in self.layer_list:
                target = self.adaptive_instance_normalization(x['r21'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad25(out)
            out = self.conv25(out)
            out = self.relu25(out)
            out = self.unpool4(out)
            out = self.reflecPad26(out)
            out = self.conv26(out)
            out = self.relu26(out)
        # r11
        if '1' <= self.start_layer[1]:
            if 'r11' == self.start_layer:
                out = x['r11']
            elif 'r11' in self.layer_list:
                target = self.adaptive_instance_normalization(x['r11'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad27(out)
            out = self.conv27(out)
        return out


class DecoderAWSC1(nn.Module):
    def __init__(self, layer_list, connect_weight=0.5):
        super(DecoderAWSC1, self).__init__()
        self.start_layer = layer_list[0]
        self.layer_list = layer_list
        self.connect_weight = connect_weight

        # decoder
        if '5' <= self.start_layer[1]:
            # r51
            self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu15 = nn.ReLU(inplace=True)

            self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
            # 28 x 28

            self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu16 = nn.ReLU(inplace=True)
            # 28 x 28

            self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu17 = nn.ReLU(inplace=True)
            # 28 x 28

            self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
            self.relu18 = nn.ReLU(inplace=True)
            # 28 x 28

        if '4' <= self.start_layer[1]:
            # r41
            self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu19 = nn.ReLU(inplace=True)
            # 28 x 28

            self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
            # 56 x 56

            self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu20 = nn.ReLU(inplace=True)
            # 56 x 56

            self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu21 = nn.ReLU(inplace=True)

            self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu22 = nn.ReLU(inplace=True)

        if '3' <= self.start_layer[1]:
            # r31
            self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu23 = nn.ReLU(inplace=True)

            self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
            # 112 X 112

            self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
            self.relu24 = nn.ReLU(inplace=True)

        if '2' <= self.start_layer[1]:
            # r21
            self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu25 = nn.ReLU(inplace=True)

            self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

            self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
            self.relu26 = nn.ReLU(inplace=True)

        if '1' <= self.start_layer[1]:
            # r11
            self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    @staticmethod
    def calc_mean(feat):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean

    def adaptive_weight_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean = self.calc_mean(style_feat)
        content_mean = self.calc_mean(content_feat)

        normalized_feat = content_feat - content_mean.expand(size)
        return normalized_feat + style_mean.expand(size)

    def forward(self, x):
        # r51
        if 'r51' == self.start_layer:
            out = self.reflecPad15(x['r51'])
            out = self.conv15(out)
            out = self.relu15(out)
            out = self.unpool(out)
            out = self.reflecPad16(out)
            out = self.conv16(out)
            out = self.relu16(out)
            out = self.reflecPad17(out)
            out = self.conv17(out)
            out = self.relu17(out)
            out = self.reflecPad18(out)
            out = self.conv18(out)
            out = self.relu18(out)
        # r41
        if '4' <= self.start_layer[1]:
            if 'r41' == self.start_layer:
                out = x['r41']
            elif 'r41' in self.layer_list:
                target = self.adaptive_weight_normalization(x['r41'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad19(out)
            out = self.conv19(out)
            out = self.relu19(out)
            out = self.unpool2(out)
            out = self.reflecPad20(out)
            out = self.conv20(out)
            out = self.relu20(out)
            out = self.reflecPad21(out)
            out = self.conv21(out)
            out = self.relu21(out)
            out = self.reflecPad22(out)
            out = self.conv22(out)
            out = self.relu22(out)
        # r31
        if '3' <= self.start_layer[1]:
            if 'r31' == self.start_layer:
                out = x['r31']
            elif 'r31' in self.layer_list:
                target = self.adaptive_weight_normalization(x['r31'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad23(out)
            out = self.conv23(out)
            out = self.relu23(out)
            out = self.unpool3(out)
            out = self.reflecPad24(out)
            out = self.conv24(out)
            out = self.relu24(out)
        # r21
        if '2' <= self.start_layer[1]:
            if 'r21' == self.start_layer:
                out = x['r21']
            elif 'r21' in self.layer_list:
                target = self.adaptive_weight_normalization(x['r21'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad25(out)
            out = self.conv25(out)
            out = self.relu25(out)
            out = self.unpool4(out)
            out = self.reflecPad26(out)
            out = self.conv26(out)
            out = self.relu26(out)
        # r11
        if '1' <= self.start_layer[1]:
            if 'r11' == self.start_layer:
                out = x['r11']
            elif 'r11' in self.layer_list:
                target = self.adaptive_weight_normalization(x['r11'], out)
                out = (1 - self.connect_weight) * out + self.connect_weight * target
            out = self.reflecPad27(out)
            out = self.conv27(out)
        return out
