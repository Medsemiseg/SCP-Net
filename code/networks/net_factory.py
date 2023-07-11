from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3, UNet_URPC, UNet_CCT, UNet_pro,BFDCNet2d_v1,DiceCENet2d_fuse,UNet_sdf
from networks.VNet import VNet, MCNet3d_v1, MCNet3d_v2, ECNet3d, DiceCENet3d,DiceCENet3d_fuse,DiceCENet3d_fuse_2

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unetsdf":
        net = UNet_sdf(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v1" and mode == "train":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "train":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "ecnet3d" and mode == "train":
        net = ECNet3d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "dicecenet3d" and mode == "train":
        net = DiceCENet3d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v1" and mode == "test":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "unet_cct" and mode == "train":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc" and mode == "train":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_pro" and mode == "train":
        net = UNet_pro(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "bfdcnet2d" and mode == "train":
        net = BFDCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "dicecenetfuse" and mode == "train":
        net = DiceCENet3d_fuse(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "dicecenetfuse_2" and mode == "train":
        net = DiceCENet3d_fuse_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "dicecenetfuse2d":
        net = DiceCENet2d_fuse(in_chns=in_chns, class_num=class_num).cuda()
    return net
