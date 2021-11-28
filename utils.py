import imagehash
from PIL import Image

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False
    
def get_hash(img):
    '''
    Get the image hash and compare it with gsch icon has 
    to make sure that the image is not gsch icon..
    '''
    hashed = imagehash.average_hash(Image.fromarray(img))
    return str(hashed)

def get_mode(lst):
    '''
    A function to get the mode from a list
    '''
    mode = max(set(lst), key=lst.count)
    return mode 