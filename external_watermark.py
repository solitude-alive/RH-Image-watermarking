import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
import numpy as np
from functools import partial


class ExtWatermark:
    """
    Watermarking Method: https://github.com/ShieldMnt/invisible-watermark
    """
    def __init__(self, method=None):
        self.message = np.random.choice([0, 1], 64)
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder('bits', 64)
        self.encoder_setup = partial(self.encoder.set_watermark, wmType='bits')
        if method is None:
            self.method = "dwtDct"
        else:
            self.method = method
        self.set_encoder()

    def set_encoder(self, mess=None):
        if mess is None:
            mess = self.message
        # print(mess)
        self.encoder_setup(content=mess)

    def encode(self, img):
        return self.encoder.encode(img, self.method)

    def decode(self, img):
        return self.decoder.decode(img, self.method)


if __name__ == "__main__":
    ext_wat_dct = ExtWatermark(method="dwtDct")
    ext_wat_dctcvd = ExtWatermark(method="dwtDctSvd")
    ext_wat = ext_wat_dct

    bgr = cv2.imread("data/demo/ori/ILSVRC2012_test_00074114.png")

    brg_enc = ext_wat.encode(bgr)

    cv2.imwrite("data/test_encoded.png", brg_enc)

    img_enc = cv2.imread("data/test_encoded.png")
    wm_dec = ext_wat.decode(img_enc)

    wm_dec = wm_dec.astype(np.int64)
    acc = np.sum(wm_dec == ext_wat.message) / len(wm_dec)
    print(acc)
