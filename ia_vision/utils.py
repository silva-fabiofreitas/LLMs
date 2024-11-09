import base64

def convert_img_in_base64(path, file=None):
    if file:
        return base64.b64encode(image).decode("utf-8")
    
    with open(path, 'rb') as file:
        image = file.read()
    return base64.b64encode(image).decode("utf-8")

