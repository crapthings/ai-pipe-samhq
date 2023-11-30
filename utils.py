from io import BytesIO
import requests
from urllib.parse import urlparse

def buff_png (image):
    buff = BytesIO()
    image.save(buff, format = 'PNG')
    buff.seek(0)
    return buff

def upload_image (url, image):
    response = requests.put(url, data = buff_png(image), headers = { 'Content-Type': 'image/png' })
    response.raise_for_status()

def extract_origin_pathname (url):
    parsed_url = urlparse(url)
    origin_pathname = parsed_url.scheme + '://' + parsed_url.netloc + parsed_url.path
    return origin_pathname
