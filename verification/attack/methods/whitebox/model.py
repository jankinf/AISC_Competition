import fnmatch
import re
import sys

import megbrain as mgb
import megskull as mgsk
import numpy as np
from meghair.utils.logconf import get_logger
from megskull.graph import FpropEnv, iter_dep_opr
from megskull.network import NetworkVisitor, RawNetworkBuilder
from megskull.opr.all import (
    BatchNormalization,
    Broadcast,
    Concat,
    ConstProvider,
    Conv2D,
    CrossEntropyLoss,
    DataProvider,
    Dropout,
    ElementwiseAffine,
    FullyConnected,
    Max,
    Pooling2D,
    Softmax,
    WarpPerspective,
    WarpPerspectiveWeightProducer,
    WeightDecay,
)
from megskull.opr.arith import ReLU
from megskull.opr.compatible.caffepool import CaffePooling2D
from megskull.utils.misc import get_2dshape
from neupeak import model as O
from neupeak.utils.cli import load_network

logger = get_logger(__name__)
sys.setrecursionlimit(10000)


def create_bn_relu(
    prefix,
    f_in,
    ksize,
    stride,
    pad,
    num_outputs,
    has_bn=True,
    has_relu=True,
    conv_name_fun=None,
    bn_name_fun=None,
):
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)

    f = Conv2D(
        conv_name,
        f_in,
        kernel_shape=ksize,
        stride=stride,
        padding=pad,
        output_nr_channel=num_outputs,
        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity(),
    )

    if has_bn:
        bn_name = "bn_" + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        f = BatchNormalization(bn_name, f)
        f._eps = 1e-9

        f = ElementwiseAffine(bn_name + "_scaleshift", f, shared_in_channels=False)

    if has_relu:
        f = ReLU(f)

    return f


def create_cconv_bn_relu(
    prefix,
    f_in,
    ksize,
    stride,
    pad,
    num_outputs,
    has_bn=True,
    has_relu=True,
    conv_name_fun=None,
    bn_name_fun=None,
):
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)

    f = Conv2D(
        conv_name,
        f_in,
        kernel_shape=ksize,
        stride=stride,
        padding=pad,
        output_nr_channel=num_outputs,
        nonlinearity=mgsk.opr.helper.elemwise_trans.Identity(),
    )

    f = Concat([-f, f], axis=1)

    if has_bn:
        bn_name = "bn_" + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        f = BatchNormalization(bn_name, f)
        f._eps = 1e-9

        f = ElementwiseAffine(bn_name + "_scaleshift", f, shared_in_channels=False)

    if has_relu:
        f = ReLU(f)

    return f


def create_crelu_bottleneck(
    prefix, f_in, stride, num_outputs1, num_outputs2, has_proj=False
):
    proj = f_in
    if has_proj:
        proj = create_bn_relu(
            prefix,
            f_in,
            ksize=1,
            stride=stride,
            pad=0,
            num_outputs=num_outputs2,
            has_bn=True,
            has_relu=False,
            conv_name_fun=lambda p: "interstellar{}_branch1".format(p),
            bn_name_fun=lambda p: "bn{}_branch1".format(p),
        )

    f = create_bn_relu(
        prefix,
        f_in,
        ksize=1,
        stride=1,
        pad=0,
        num_outputs=num_outputs1,
        has_bn=True,
        has_relu=True,
        conv_name_fun=lambda p: "interstellar{}_branch2a".format(p),
        bn_name_fun=lambda p: "bn{}_branch2a".format(p),
    )

    f = create_cconv_bn_relu(
        prefix,
        f,
        ksize=3,
        stride=stride,
        pad=1,
        num_outputs=num_outputs1,
        has_bn=True,
        has_relu=True,
        conv_name_fun=lambda p: "interstellar{}_branch2b".format(p),
        bn_name_fun=lambda p: "bn{}_branch2b".format(p),
    )

    f = create_bn_relu(
        prefix,
        f,
        ksize=1,
        stride=1,
        pad=0,
        num_outputs=num_outputs2,
        has_bn=True,
        has_relu=False,
        conv_name_fun=lambda p: "interstellar{}_branch2c".format(p),
        bn_name_fun=lambda p: "bn{}_branch2c".format(p),
    )

    f = f + proj

    return ReLU(f)


def name_fun(suffix):
    return {
        "conv_name_fun": lambda p: p + "_branch" + suffix,
        "bn_name_fun": lambda p: "bn_" + p + "_branch" + suffix,
    }


def create_inception_res(prefix, f_in, dims, stride, pool_type):
    assert len(dims) == 9

    f_1x1 = create_bn_relu(
        prefix,
        f_in,
        ksize=1,
        stride=stride,
        pad=0,
        num_outputs=dims[0],
        **name_fun("1x1")
    )
    f_1x1_3x3 = create_bn_relu(
        prefix,
        f_in,
        ksize=1,
        stride=1,
        pad=0,
        num_outputs=dims[1],
        **name_fun("1x1_3x3a")
    )
    f_1x1_3x3 = create_bn_relu(
        prefix,
        f_1x1_3x3,
        ksize=3,
        stride=stride,
        pad=1,
        num_outputs=dims[2],
        **name_fun("1x1_3x3b")
    )
    f_1x1_d3x3 = create_bn_relu(
        prefix,
        f_in,
        ksize=1,
        stride=1,
        pad=0,
        num_outputs=dims[3],
        **name_fun("1x1_duo3x3a")
    )
    f_1x1_d3x3 = create_bn_relu(
        prefix,
        f_1x1_d3x3,
        ksize=3,
        stride=stride,
        pad=1,
        num_outputs=dims[4],
        **name_fun("1x1_duo3x3b")
    )
    f_1x1_d3x3 = create_bn_relu(
        prefix,
        f_1x1_d3x3,
        ksize=3,
        stride=1,
        pad=1,
        num_outputs=dims[5],
        **name_fun("1x1_duo3x3c")
    )
    f_concat = [f_1x1, f_1x1_3x3, f_1x1_d3x3]

    if dims[6] > 0:
        f_pool = Pooling2D(
            "pool" + prefix, f_in, window=3, stride=stride, padding=1, mode=pool_type
        )
        f_pool = create_bn_relu(
            prefix,
            f_pool,
            ksize=1,
            stride=1,
            pad=0,
            num_outputs=dims[6],
            **name_fun("pool")
        )
        f_concat = [f_pool] + f_concat

    f_concat = Concat(f_concat, axis=1)
    f_res = create_bn_relu(
        prefix,
        f_concat,
        ksize=1,
        stride=1,
        pad=0,
        num_outputs=dims[7],
        has_bn=True,
        has_relu=False,
        **name_fun("red")
    )

    if dims[8] > 0:
        f_shortcut = create_bn_relu(
            prefix,
            f_in,
            ksize=1,
            stride=stride,
            pad=0,
            num_outputs=dims[8],
            has_bn=True,
            has_relu=False,
            **name_fun("proj")
        )
    else:
        f_shortcut = f_in

    return ReLU(f_res + f_shortcut)


def create_resize(name, f_in, f_ref, ratio_x, ratio_y):
    trans = np.array(
        [[ratio_x, 0, 0], [0, ratio_y, 0], [0, 0, 1]], dtype=np.float32
    ).reshape([1, 3, 3])
    trans = ConstProvider(trans)
    trans = Broadcast(trans, Concat([f_in.shape[0], 3, 3], axis=0))
    output_shape = f_ref.shape[2:]
    return WarpPerspective(
        name,
        f_in,
        trans,
        output_shape,
        border_mode=mgb.opr_param_defs.WarpPerspective.BorderMode.CONSTANT,
        border_val=0,
    )


def BN_ResNet(input_image, mode="rgb"):
    f = create_cconv_bn_relu(
        "{}_conv1".format(mode), input_image, ksize=7, stride=2, pad=3, num_outputs=16
    )
    f = Pooling2D("{}_pool1".format(mode), f, window=3, stride=2, padding=1, mode="MAX")

    f = create_crelu_bottleneck("{}_2a".format(mode), f, 1, 24, 64, has_proj=True)
    f = create_crelu_bottleneck("{}_2b".format(mode), f, 1, 24, 64)
    f = create_crelu_bottleneck("{}_2c".format(mode), f, 1, 24, 64)

    f = create_crelu_bottleneck("{}_3a".format(mode), f, 2, 48, 128, has_proj=True)
    f = create_crelu_bottleneck("{}_3b".format(mode), f, 1, 48, 128)
    f = create_crelu_bottleneck("{}_3c".format(mode), f, 1, 48, 128)
    f = create_crelu_bottleneck("{}_3d".format(mode), f, 1, 48, 128)
    f_28 = f

    f = create_inception_res(
        "{}_4a".format(mode), f, [64, 48, 128, 24, 48, 48, 128, 256, 256], 2, "MAX"
    )
    f = create_inception_res(
        "{}_4b".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4c".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4d".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4e".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4f".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4g".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4h".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4i".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4j".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4k".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_4l".format(mode), f, [64, 64, 128, 24, 48, 48, 0, 256, 0], 1, "MAX"
    )
    f_14 = f

    f = create_inception_res(
        "{}_5a".format(mode), f, [64, 96, 192, 32, 64, 64, 128, 512, 512], 2, "MAX"
    )
    f = create_inception_res(
        "{}_5b".format(mode), f, [64, 96, 192, 32, 64, 64, 0, 512, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_5c".format(mode), f, [64, 96, 192, 32, 64, 64, 0, 512, 0], 1, "MAX"
    )
    f = create_inception_res(
        "{}_5d".format(mode), f, [64, 96, 192, 32, 64, 64, 0, 512, 0], 1, "MAX"
    )

    f_down = Pooling2D(
        "{}_downsample".format(mode), f_28, window=3, stride=2, padding=1, mode="MAX"
    )
    f_up = create_resize("{}_upsample".format(mode), f, f_14, 0.5, 0.5)
    f = Concat([f_down, f_14, f_up], axis=1)

    f = create_bn_relu(
        "{}_final_bn".format(mode),
        f,
        ksize=1,
        stride=1,
        pad=0,
        num_outputs=512,
        has_bn=True,
        has_relu=True,
        conv_name_fun=lambda p: "{}_conv_feature".format(p),
        bn_name_fun=lambda p: "{}_bn_conv_feature".format(p),
    )

    # recommend feature map (f) from here
    f = Pooling2D("last-pool-1", f, window=2, stride=2, mode="max")
    f = create_bn_relu(
        "cbnre-prefc-1",
        f,
        ksize=3,
        stride=1,
        pad=0,
        num_outputs=128,
        **name_fun("cbnre-prefc-1")
    )

    x = O.global_pooling_v2(f, mode="max")
    x = O.fully_connected("fct-2-prev", x, output_dim=64)
    x = O.fully_connected(
        "fct-2", x, output_dim=2, nonlinearity=O.nonlinearity.Identity()
    )
    x = O.softmax(x)

    return x


def make_network(config):
    rgb_input = O.input("data", shape=config.DATA.INPUT_SHAPE)
    label = O.input("label", shape=(config.DATA.BATCH_SIZE,), dtype="int32")

    pred = BN_ResNet(rgb_input, "rgb")
    
    losses = dict()
    loss_xent = O.cross_entropy(pred, label, name="loss_xent")
    losses["loss_xent"] = loss_xent

    weight_decay = 1e-5
    pattern = re.compile("|".join(map(fnmatch.translate, ["*conv*:W", "*fc*:W"])))
    loss_weight_decay = weight_decay * sum(
        (opr**2).sum() for opr in iter_dep_opr(loss_xent) if pattern.match(opr.name)
    )

    losses["loss_weight_decay"] = loss_weight_decay

    loss = sum(losses.values())

    O.utils.hint_loss_subgraph(list(losses.values()), loss)

    # build network
    network = RawNetworkBuilder(inputs=rgb_input, outputs=pred, loss=loss)
    extra_outputs = losses.copy()
    extra_outputs.update(
        misclassify=O.misclassify(pred, label),
        misclassify0=O.misclassify(pred, label, incl_mask=O.eq(label, 0)),
        misclassify1=O.misclassify(pred, label, incl_mask=O.eq(label, 1)),
        accuracy=O.accuracy(pred, label),
    )
    network.extra["extra_outputs"] = extra_outputs
    network.extra["extra_config"] = {
        "monitor_vars": ["misclassify", "misclassify0", "misclassify1", "accuracy"],
    }

    O.param_init.set_opr_states_from_network(
        network.loss_visitor.all_oprs,
        "/data/flashfmp-feature/feature_model/epoch.13020.0_0001.pickle",
        check_shape=False,
    )

    return network


if __name__ == "__main__":
    import neupeak.utils.inference as inf

    from config import cfg

    cfg.defrost()
    cfg.DATA.BATCH_SIZE = 8
    cfg.freeze()

    net = make_network(cfg)
    f = inf.Function(inf.get_fprop_env(fast_run=False))
    mgb.config.set_comp_graph_option(f.comp_graph, "log_level", 0)
    pred_func = f.compile(net.outputs)

    img = np.random.randn(*cfg.DATA.INPUT_SHAPE).astype(np.float32)
    pred = pred_func(img)

    print(pred[0].shape)
