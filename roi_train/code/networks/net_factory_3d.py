from networks.vnet import VNet


def net_factory_3d(net_type="vnet", in_chns=1, class_num=2, has_dropout=False):
    if net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True)
    # elif net_type == "unet_3D_dt":
    #     net = unet_3D_dt(n_classes=class_num, in_channels=in_chns)
    else:
        net = None
    return net