import cv2
import itertools
import numpy as np
import random
import torch
import torchgeometry


def transform_net(encoded_image, global_step):
    rnd_bri_ramp = 1000
    rnd_bri = 0.3 * 0.6
    rnd_hue_ramp = 1000     # default is 1000
    rnd_hue = 0.1    # default is 0.1
    batch_size = encoded_image.shape[0]
    jpeg_quality_ramp = 1000
    jpeg_quality = 50
    rnd_noise_ramp = 1000
    rnd_noise = 0.02
    contrast_low = 0.8
    contrast_ramp = 1000
    contrast_high = 1.2
    rnd_sat_ramp = 1000
    rnd_sat = 0.8 * 0.6
    no_jpeg = None

    global_step = global_step * 20  # max is 1000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoded_image = encoded_image.to(device)
    img_size = encoded_image.shape[2]

    encoded_image = tansform_geometry(encoded_image, img_size, img_size, global_step, batch_size, device)

    encoded_image = encoded_image.permute(0, 2, 3, 1)
    encoded_image = encoded_image / 2 + 0.5

    # sh = tf.shape(encoded_image)

    def ramp_fn(ramp, noise_step=global_step):
        """
        Note the return value is not greater than 1.
        """
        return torch.tensor(noise_step).float() / ramp

    rnd_bri = ramp_fn(rnd_bri_ramp) * rnd_bri
    rnd_hue = ramp_fn(rnd_hue_ramp) * rnd_hue
    rnd_brightness = get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size).to(device)

    jpeg_quality = 100. - torch.rand([]) * ramp_fn(jpeg_quality_ramp) * (100. - jpeg_quality)
    if jpeg_quality < 50:
        jpeg_factor = 5000. / jpeg_quality
    else:
        jpeg_factor = 200. - jpeg_quality * 2
    jpeg_factor = jpeg_factor / 100. + .0001

    rnd_noise = torch.rand([]) * ramp_fn(rnd_noise_ramp) * rnd_noise

    contrast_low = 1. - (1. - contrast_low) * ramp_fn(contrast_ramp)
    contrast_high = 1. + (contrast_high - 1.) * ramp_fn(contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand([]) * ramp_fn(rnd_sat_ramp) * rnd_sat

    # blur
    f = random_blur_kernel(probs=[.25, .25], N_blur=7,
                           sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3)
    conv_blur = torch.nn.Conv2d(3, 3, 7, 1, padding=3).to(device)
    conv_blur.weight.data = f.to(device)
    encoded_image = encoded_image.permute(0, 3, 1, 2)
    encoded_image = conv_blur(encoded_image)
    encoded_image = encoded_image.permute(0, 2, 3, 1)

    noise = torch.normal(mean=0.0, std=rnd_noise, size=encoded_image.shape).to(device)  # normal distribution
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, min=0, max=1)

    contrast_scale = torch.rand(encoded_image.shape[0]) * (contrast_params[1] - contrast_params[0]) + contrast_params[0]
    contrast_scale = torch.reshape(contrast_scale, shape=[encoded_image.shape[0], 1, 1, 1])

    encoded_image = encoded_image * contrast_scale.to(device)
    encoded_image = encoded_image + rnd_brightness.to(device)
    encoded_image = torch.clamp(encoded_image, min=0, max=1)

    encoded_image_lum = (torch.sum(encoded_image * torch.tensor([.3, .6, .1]).to(device), dim=3)).unsqueeze(3)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    encoded_image = torch.reshape(encoded_image, [-1, img_size, img_size, 3])
    if not no_jpeg:
        encoded_image = jpeg_compress_decompress(encoded_image, rounding=round_only_at_0,
                                                 factor=jpeg_factor, downsample_c=True)

    encoded_image = encoded_image.permute(0, 3, 1, 2)
    encoded_image = (encoded_image - 0.5) * 2
    # encoded_image = torch.tensor(encoded_image.numpy()).to('cuda')

    return encoded_image


def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = (torch.stack(torch.meshgrid(
        torch.tensor(range(0, N_blur)), torch.tensor(range(0, N_blur))), -1)).float() - (.5 * (N - 1))
    # coords = tf.to_float(coords)
    manhat = torch.sum(torch.abs(coords), -1)

    # nothing, default

    vals_nothing = (manhat < .5).float()

    # gauss

    sig_gauss = torch.rand([]) * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords ** 2, -1) / 2. / sig_gauss ** 2)

    # line

    theta = torch.rand([]) * 2. * np.pi
    v = torch.tensor([torch.cos(theta), torch.sin(theta)])
    dists = torch.sum(coords * v, -1)

    sig_line = torch.rand([]) * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    w_line = torch.rand([]) * (.5 * (N - 1) + .1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists ** 2 / 2. / sig_line ** 2) * (manhat < w_line).float()

    t = torch.rand([])
    vals = vals_nothing
    if t < probs[0] + probs[1]:
        vals = vals_line
    if t < probs[0]:
        vals = vals_gauss

    v = vals / torch.sum(vals)
    z = torch.zeros_like(v)
    f = torch.reshape(torch.stack([v, z, z, z, v, z, z, z, v], -1), [N, N, 3, 3])
    f = f.permute(2, 3, 0, 1)  # f.shape = (3, 3, N, N)

    return f


# def get_rand_transform_matrix(image_size, d, batch_size):
#     Ms = np.zeros((batch_size, 2, 8))
#
#     for i in range(batch_size):
#         tl_x = random.uniform(-d, d)  # Top left corner, top
#         tl_y = random.uniform(-d, d)  # Top left corner, left
#         bl_x = random.uniform(-d, d)  # Bot left corner, bot
#         bl_y = random.uniform(-d, d)  # Bot left corner, left
#         tr_x = random.uniform(-d, d)  # Top right corner, top
#         tr_y = random.uniform(-d, d)  # Top right corner, right
#         br_x = random.uniform(-d, d)  # Bot right corner, bot
#         br_y = random.uniform(-d, d)  # Bot right corner, right
#
#         rect = np.array([
#             [tl_x, tl_y],
#             [tr_x + image_size, tr_y],
#             [br_x + image_size, br_y + image_size],
#             [bl_x, bl_y + image_size]], dtype="float32")
#
#         dst = np.array([
#             [0, 0],
#             [image_size, 0],
#             [image_size, image_size],
#             [0, image_size]], dtype="float32")
#
#         M = cv2.getPerspectiveTransform(rect, dst)
#         M_inv = np.linalg.inv(M)
#         Ms[i, 0, :] = M_inv.flatten()[:8]
#         Ms[i, 1, :] = M.flatten()[:8]
#     return Ms


def get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size):
    rnd_hue = torch.rand(batch_size, 1, 1, 3) * rnd_hue - rnd_hue  # Uniform distribution [0, 1)
    rnd_brightness = torch.rand(batch_size, 1, 1, 1) * rnd_bri - rnd_bri   # Uniform distribution [0, 1)
    return rnd_hue + rnd_brightness


## Differentiable JPEG, Source - https://github.com/rshin/differentiable-jpeg/blob/master/jpeg-tensorflow.ipynb

# 1. RGB -> YCbCr
# https://en.wikipedia.org/wiki/YCbCr


def rgb_to_ycbcr_jpeg(image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    matrix = torch.tensor(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=torch.float32).T.to(device)
    shift = torch.tensor([0., 128., 128.]).to(device)

    result = torch.tensordot(image, matrix, dims=1) + shift
    result.view(image.shape)
    return result


# 2. Chroma subsampling
def downsampling_420(image):
    # input: batch x height x width x 3
    # output: tuple of length 3
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    y, cb, cr = image[:, :, :, 0].unsqueeze(3), image[:, :, :, 1].unsqueeze(3), image[:, :, :, 2].unsqueeze(3)
    avg_pool = torch.nn.AvgPool2d(2, stride=2)

    cb = cb.permute(0, 3, 1, 2)
    cr = cr.permute(0, 3, 1, 2)
    cb = avg_pool(cb)
    cr = avg_pool(cr)
    cb = cb.permute(0, 2, 3, 1)
    cr = cr.permute(0, 2, 3, 1)

    return (torch.squeeze(
        y, dim=-1), torch.squeeze(
        cb, dim=-1), torch.squeeze(
        cr, dim=-1))


# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
def image_to_patches(image):
    # input: batch x h x w
    # output: batch x h*w/64 x h x w
    k = 8
    height, width = image.shape[1:3]
    batch_size = image.shape[0]
    image_reshaped = torch.reshape(image, [batch_size, height // k, k, -1, k])
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return torch.reshape(image_transposed, [batch_size, -1, k, k])


# 4. DCT
def dct_8x8_ref(image):
    image = image - 128
    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        result[u, v] = value
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    return result * scale


def dct_8x8(image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    result = torch.tensor(scale).to(device) * torch.tensordot(image, torch.tensor(tensor).to(device), dims=2)
    # result.set_shape(image.shape.as_list())
    return result


# 5. Quantizaztion
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T


def y_quantize(image, rounding, factor=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = image / (torch.tensor(y_table).to(device) * factor)
    image = rounding(image)
    return image


def c_quantize(image, rounding, factor=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = image / (torch.tensor(c_table).to(device) * factor)
    image = rounding(image)
    return image


# -5. Dequantization
def y_dequantize(image, factor=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return image * (torch.tensor(y_table).to(device) * factor)


def c_dequantize(image, factor=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return image * (torch.tensor(c_table).to(device) * factor)


# -4. Inverse DCT
def idct_8x8_ref(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha

    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        result[u, v] = value
    return result * 0.25 + 128


def idct_8x8(image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * torch.tensor(alpha).to(device)

    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    result = torch.tensor(0.25).to(device) * torch.tensordot(image.to(torch.float32), torch.tensor(tensor).to(device),
                                                             dims=2) + torch.tensor(128).to(device)
    # result.set_shape(image.shape.as_list())
    return result


# -3. Block joining
def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    batch_size = patches.shape[0]
    image_reshaped = torch.reshape(patches,
                                   [batch_size, height // k, width // k, k, k])
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return torch.reshape(image_transposed, [batch_size, height, width])


# -2. Chroma upsampling
def upsampling_420(y, cb, cr):
    # input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # output:
    #   image: batch x height x width x 3
    def repeat(x, k=2):
        height, width = x.shape[1:3]
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, k, k)
        x = torch.reshape(x, [-1, height * k, width * k])
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return torch.stack((y, cb, cr), dim=-1)


# -1. YCbCr -> RGB
def ycbcr_to_rgb(image):
    matrix = np.array(
        [[298.082, 0, 408.583], [298.082, -100.291, -208.120],
         [298.082, 516.412, 0]],
        dtype=np.float32).T / 256
    shift = [-222.921, 135.576, -276.836]

    result = torch.tensordot(image, matrix, dims=1) + shift
    result.set_shape(image.shape.as_list())
    return result


def ycbcr_to_rgb_jpeg(image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    matrix = np.array(
        [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
        dtype=np.float32).T
    shift = torch.tensor([0, -128, -128]).to(device)

    result = torch.tensordot(image + shift, torch.tensor(matrix).to(device), dims=1)
    # result.set_shape(image.shape.as_list())
    return result


def diff_round(x):
    return torch.round(x) + (x - torch.round(x)) ** 3


def round_only_at_0(x):
    cond = (torch.abs(x) < 0.5).to(torch.float32)
    return cond * (x ** 3) + (1 - cond) * x


def jpeg_compress_decompress(image,
                             downsample_c=True,
                             rounding=diff_round,
                             factor=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image *= 255
    height, width = image.shape[1:3]
    orig_height, orig_width = height, width
    # if height % 16 != 0 or width % 16 != 0:
    #     # Round up to next multiple of 16
    #     height = ((height - 1) // 16 + 1) * 16
    #     width = ((width - 1) // 16 + 1) * 16
    #
    #     vpad = height - orig_height
    #     wpad = width - orig_width
    #     top = vpad // 2
    #     bottom = vpad - top
    #     left = wpad // 2
    #     right = wpad - left
    #
    #     # image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
    #     image = tf.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'SYMMETRIC')
    assert height % 16 == 0

    # "Compression"
    image = rgb_to_ycbcr_jpeg(image)
    if downsample_c:
        y, cb, cr = downsampling_420(image)
    else:
        y, cb, cr = torch.split(image, 3, dim=3)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding,
                          factor) if k in ('cb', 'cr') else y_quantize(
            comp, rounding, factor)
        components[k] = comp

    # "Decompression"
    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(
            comp, factor)
        comp = idct_8x8(comp)
        if k in ('cb', 'cr'):
            if downsample_c:
                comp = patches_to_image(comp, int(height / 2), int(width / 2))
            else:
                comp = patches_to_image(comp, height, width)
        else:
            comp = patches_to_image(comp, height, width)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = torch.stack((y, cb, cr), dim=-1)
    image = ycbcr_to_rgb_jpeg(image)

    # Crop to original size
    # if orig_height != height or orig_width != width:
    #     # image = image[:, top:-bottom, left:-right]
    #     image = image[:, :-vpad, :-wpad]

    # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
    #    min_value = tf.minimum(tf.reduce_min(image), 0.)
    #    max_value = tf.maximum(tf.reduce_max(image), 255.)
    #    value_range = max_value - min_value
    #    image = 255 * (image - min_value) / value_range
    image = torch.minimum(torch.tensor(255.).to(device), torch.maximum(torch.tensor(0.).to(device), image))
    image /= 255

    return image


def quality_to_factor(quality):
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 3, 3))
    for i in range(batch_size):
        tl_x = random.uniform(-d, d)  # Top left corner, top
        tl_y = random.uniform(-d, d)  # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)  # Bot left corner, left
        tr_x = random.uniform(-d, d)  # Top right corner, top
        tr_y = random.uniform(-d, d)  # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)  # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size]], dtype="float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i, 0, :, :] = M_inv
        Ms[i, 1, :, :] = M
    Ms = torch.from_numpy(Ms).float()

    return Ms


def tansform_geometry(image, width, height, global_step, batch_size, device):
    rnd_tran = min(0.1 * global_step / 1000, 0.1)
    rnd_tran = np.random.uniform() * rnd_tran
    M = get_rand_transform_matrix(width, np.floor(width * rnd_tran), batch_size).to(device)

    borders_list = ["no_edge", "black", "random", "white", "image", None]
    borders = random.choice(borders_list)
    # borders = "random"
    # print(borders)

    if borders == 'no_edge':
        # print(0)
        pass
    elif borders == 'black':
        image = torchgeometry.warp_perspective(image, M[:, 0, :, :], dsize=(width, height), flags='bilinear')
    elif borders == 'random':
        mask = torchgeometry.warp_perspective(torch.ones_like(image), M[:, 0, :, :], dsize=(width, height),
                                              flags='bilinear')
        image = torchgeometry.warp_perspective(image, M[:, 0, :, :], dsize=(width, height), flags='bilinear')
        ch = 3
        image += (1 - mask) * torch.ones_like(image) * torch.rand([1, ch, 1, 1]).to(device)
    elif borders == 'white':
        mask = torchgeometry.warp_perspective(torch.ones_like(image), M[:, 0, :, :], dsize=(width, height),
                                              flags='bilinear')
        image = torchgeometry.warp_perspective(image, M[:, 0, :, :], dsize=(width, height), flags='bilinear')
        image += (1 - mask) * torch.ones_like(image)
    elif borders == 'image':
        mask = torchgeometry.warp_perspective(torch.ones_like(image), M[:, 0, :, :], dsize=(width, height),
                                              flags='bilinear')
        image = torchgeometry.warp_perspective(image, M[:, 0, :, :], dsize=(width, height), flags='bilinear')
        image += (1 - mask) * torch.roll(image, 1, 0)

    return image


def transform_test(encoded_image, global_step):
    rnd_bri_ramp = 1000
    rnd_bri = 0.3
    rnd_hue_ramp = 1000    # default is 1000
    rnd_hue = 0.1  # default is 0.1
    batch_size = encoded_image.shape[0]
    jpeg_quality_ramp = 1000
    jpeg_quality = 50
    rnd_noise_ramp = 1000
    rnd_noise = 0.02
    contrast_low = 0.8
    contrast_ramp = 1000
    contrast_high = 1.5
    rnd_sat_ramp = 1000
    rnd_sat = 0.5
    no_jpeg = None

    if global_step >= 50:
        norm = torch.nn.InstanceNorm2d(3)
        encoded_image = (encoded_image + norm(encoded_image)) / 2

    global_step = global_step * 10 if (global_step > 1) else global_step

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoded_image = encoded_image.to(device)
    img_size = encoded_image.shape[2]

    encoded_image = tansform_geometry(encoded_image, img_size, img_size, global_step, batch_size, device)

    encoded_image = encoded_image.permute(0, 2, 3, 1)
    encoded_image = encoded_image / 2 + 0.5

    # sh = tf.shape(encoded_image)

    ramp_fn = lambda ramp: torch.minimum(torch.tensor(global_step).float() / ramp, torch.tensor(1.))

    rnd_bri = ramp_fn(rnd_bri_ramp) * rnd_bri
    rnd_hue = ramp_fn(rnd_hue_ramp) * rnd_hue
    rnd_brightness = get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size).to(device)

    jpeg_quality = 100. - torch.rand([]) * ramp_fn(jpeg_quality_ramp) * (100. - jpeg_quality)
    if jpeg_quality < 50:
        jpeg_factor = 5000. / jpeg_quality
    else:
        jpeg_factor = 200. - jpeg_quality * 2
    jpeg_factor = jpeg_factor / 100. + .0001

    rnd_noise = torch.rand([]) * ramp_fn(rnd_noise_ramp) * rnd_noise

    contrast_low = 1. - (1. - contrast_low) * ramp_fn(contrast_ramp)
    contrast_high = 1. + (contrast_high - 1.) * ramp_fn(contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    # rnd_sat = torch.rand([]) * ramp_fn(rnd_sat_ramp) * rnd_sat

    # blur
    f = random_blur_kernel(probs=[.25, .25], N_blur=7,
                           sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3)
    conv_blur = torch.nn.Conv2d(3, 3, 7, 1, padding=3).to(device)
    conv_blur.weight.data = f.to(device)
    encoded_image = encoded_image.permute(0, 3, 1, 2)
    encoded_image = conv_blur(encoded_image)
    encoded_image = encoded_image.permute(0, 2, 3, 1)

    noise = torch.normal(mean=0.0, std=rnd_noise, size=encoded_image.shape).to(device)  # normal distribution
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, min=0, max=1)

    contrast_scale = torch.rand(encoded_image.shape[0]) * (contrast_params[1] - contrast_params[0]) + contrast_params[0]
    contrast_scale = torch.reshape(contrast_scale, shape=[encoded_image.shape[0], 1, 1, 1])

    encoded_image = encoded_image * contrast_scale.to(device)
    encoded_image = encoded_image + rnd_brightness.to(device)
    encoded_image = torch.clamp(encoded_image, min=0, max=1)

    encoded_image_lum = (torch.sum(encoded_image * torch.tensor([.3, .6, .1]).to(device), dim=3)).unsqueeze(3)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    encoded_image = torch.reshape(encoded_image, [-1, img_size, img_size, 3])
    if not no_jpeg:
        encoded_image = jpeg_compress_decompress(encoded_image, rounding=round_only_at_0,
                                                 factor=jpeg_factor, downsample_c=True)

    encoded_image = encoded_image.permute(0, 3, 1, 2)
    encoded_image = (encoded_image - 0.5) * 2
    # encoded_image = torch.tensor(encoded_image.numpy()).to('cuda')

    return encoded_image


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "6" 	# 指定第1块GPU
#
# device = torch.device('cuda')
# img_test = torch.randn((32, 3, 256, 256)).to(device)
# result = transform_net(img_test, 100)
# print(result.shape)
