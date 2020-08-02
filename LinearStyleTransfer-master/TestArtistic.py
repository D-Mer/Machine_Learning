import os
import torch
import argparse
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.models import encoder3, encoder4, encoder5
from libs.models import decoder3, decoder4, decoder5
from time import time


def print_time(start_time, msg=''):
    now = int(time() - start_time)
    h = now // 3600
    now -= h * 3600
    m = now // 60
    now -= m * 60
    print('%2d:%2d:%2d %s' % (h, m, now, msg))


if __name__ == '__main__':
    st = time()
    print_time(st, 'init...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                        help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                        help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                        help='pre-trained model path')
    parser.add_argument("--stylePath", default="data/style/",
                        help='path to style image')
    parser.add_argument("--contentPath", default="data/content/",
                        help='path to frames')
    parser.add_argument("--outf", default="Artistic/",
                        help='path to transferred images')
    parser.add_argument("--batchSize", type=int, default=1,
                        help='batch size')
    parser.add_argument('--loadSize', type=int, default=256,
                        help='scale image size')
    parser.add_argument('--fineSize', type=int, default=256,
                        help='crop image size')
    # parser.add_argument("--layer", default="r41",
    #                     help='which features to transfer, either r31 or r41')
    parser.add_argument("--layer", default="r41",
                        help='which features to transfer, either r31 or r41')

    ################# PREPARATIONS #################
    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)

    os.makedirs(opt.outf, exist_ok=True)
    cudnn.benchmark = True

    ################# DATA #################
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize, test=True)
    content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                                 batch_size=opt.batchSize,
                                                 shuffle=False,
                                                 num_workers=1)
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize, test=True)
    style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=1)

    ################# MODEL #################
    # 选择对应的特征层
    if (opt.layer == 'r31'):
        vgg = encoder3()
        dec = decoder3()
    elif (opt.layer == 'r41'):
        vgg = encoder4()
        dec = decoder4()
    matrix = MulLayer(opt.layer)
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))

    ################# GLOBAL VARIABLE #################
    # 设置输入网络的原图像张量，这里的size后面会被resize为原图的长宽比，fineSize是长宽里最小的值
    contentV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
    styleV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

    ################# GPU  #################
    if (opt.cuda):
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
        contentV = contentV.cuda()
        styleV = styleV.cuda()

    start_trans_time = time()
    count = 0
    print_time(st, 'finish init...')
    for ci, (content, contentName) in enumerate(content_loader):
        contentName = contentName[0]
        # 将内容图resize为指定大小(最小边为256)
        contentV.resize_(content.size()).copy_(content)
        for sj, (style, styleName) in enumerate(style_loader):
            count += 1
            start_time = time()
            print_time(st, 'start trans content: %s , style: %s' % (contentName, styleName[0]))
            styleName = styleName[0]
            # 将风格图resize为指定大小(最小边为256)
            styleV.resize_(style.size()).copy_(style)

            # forward
            with torch.no_grad():
                # 分别计算内容图和风格图的feature map
                sF = vgg(styleV)
                cF = vgg(contentV)

                # 进行迁移计算
                if (opt.layer == 'r41'):
                    feature, transmatrix = matrix(cF[opt.layer], sF[opt.layer])
                else:
                    feature, transmatrix = matrix(cF, sF)

                # 迁移计算后解码生成结果图
                transfer = dec(feature)

            transfer = transfer.clamp(0, 1)
            vutils.save_image(transfer, '%s/%s_%s_%s.png' % (opt.outf, contentName, styleName, opt.layer), normalize=True,
                              scale_each=True, nrow=opt.batchSize)
            print('Transferred image saved at %s%s_%s.png' % (opt.outf, contentName, styleName))
            end_time = time()
            print_time(st, 'finish trans content: %s , style: %s , using %d ms' %
                       (contentName, styleName, (end_time - start_time) * 1000))
    finish_trans_time = time()
    print_time(st, 'finish %d trans, using %d ms' % (count, (finish_trans_time - start_trans_time) * 1000))
    print_time(st, 'fine img size : %d' % opt.fineSize)
    print_time(st, 'avg time for trans one img: %.3f ms' % ((finish_trans_time - start_trans_time) * 1000 / count))