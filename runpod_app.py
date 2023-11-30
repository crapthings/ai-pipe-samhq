import requests
from PIL import ImageOps
from diffusers.utils import load_image
from transparent_background import Remover
import runpod

from utils import buff_png, upload_image, extract_origin_pathname

remover = Remover()

def run (job):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')

        input_url = _input.get('input_url')
        upload_url = _input.get('upload_url')
        get_mask = _input.get('get_mask')
        background_color = _input.get('background_color')
        get_overlay = _input.get('get_overlay')
        invert_mask = _input.get('invert_mask')

        # move later
        input_image = load_image(input_url)
        get_mask = 'map' if get_mask else 'rgba'

        if type(background_color) == str:
            get_mask = background_color

        if get_overlay is True:
            get_mask = 'overlay'

        output_image = remover.process(
            input_image,
            type = get_mask
        )

        if background_color is None and get_overlay is None and invert_mask is True:
            output_image = ImageOps.invert(output_image)

        # output
        output_url = extract_origin_pathname(upload_url)
        output = { 'output_url': output_url }

        # payload
        payload = job.copy()
        payload['output'] = output

        upload_image(upload_url, output_image)

        return output
    # caught http[s] error
    except requests.exceptions.RequestException as e:
        return { 'error': e.args[0] }

runpod.serverless.start({ 'handler': run })
