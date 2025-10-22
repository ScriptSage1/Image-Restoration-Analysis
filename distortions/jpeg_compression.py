import cv2 as cv

def apply_JpegCompression(image, quality = 100):
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]

    _, encoded_img = cv.imencode('.jpg', cv.cvtColor(image, cv.COLOR_RGB2BGR), encode_param)

    decoded_img = cv.imdecode(encoded_img, cv.IMREAD_COLOR)

    decoded_img = cv.cvtColor(decoded_img , cv.COLOR_BGR2RGB)

    return decoded_img
