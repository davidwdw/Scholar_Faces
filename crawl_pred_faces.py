import sys,os,cv2,pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib.request
from deepface import DeepFace
from utils import isnotebook
from utils import get_hash
from utils import get_mode

# check if this is a notebook
#---------------------------#

if isnotebook():
    chunk=17
    to_stop_limit=None
else:
    parser=argparse.ArgumentParser()
    parser.add_argument('--chunk',type=int,default=0,
                        choices=list(range(20)))
    chunk=parser.parse_args().chunk
    to_stop_limit=None
    
# read the csv containing gsch ids
#--------------------------------#

# this is from Dashun's hot streak paper
df = pd.read_csv('./gs_name_profile.txt',
                 sep='\t',
                 header=None,
                 names=['index','name','work'],
                 index_col='index')

# divide into chunks to send to slurm separately
df = np.array_split(df,20)[chunk]

# init the dst folder
dst_dir = './data/'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

# init functions
#--------------#

# google scholar image request function
def get_img_from_gsch_id(gsch_id):
    '''
    Request image from GScholar using the person's GScholar ID
    Return None if request failed. 
    '''
    google_url = 'https://scholar.googleusercontent.com/citations'
    default_url = google_url+'?view_op=medium_photo&user={}&citpid=2'
    url = default_url.format(gsch_id)
    
    try:
        img = urllib.request.urlopen(url).read()
        img = cv2.imdecode(np.frombuffer(img,np.uint8),-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
    except:
        img = None
    
    return img

# predict demographics function
def get_demo(dst_fp):
    '''
    Get the demographics from a facial image
    Use multiple models and get the mode from their 
    outputs..
    If face not detected, will return empty dict.
    '''
    backends = ['opencv','dlib','retinaface']
    demos = {}
    
    for backend in backends:
        try:
            demo = DeepFace.analyze(dst_fp,
                                    detector_backend=backend)
            temp_ = demo['emotion']
            temp_.update({'age':demo['age']})
            gender = 0 if demo['gender']=='Man' else 1
            temp_.update({'gender':gender})
            temp_.update(demo['race'])
            demos[backend] = temp_
            
        except:
            pass

    if demos != {}:
        
        demos = pd.DataFrame.from_dict(
            demos,orient='index').mean().to_dict()
        
    return demos

# this is to download the image
#-----------------------------#

count = 1; total = len(df)

for i,v in df.iterrows():
    
    print(f'Downloading image : {count}/{total} | {i}',
          end='\r');count+=1
    
    dst_fp = dst_dir+i+'/img.jpg'
    
    if not os.path.exists(dst_fp):
        
        # download the image
        img = get_img_from_gsch_id(i)
        
        if img is not None:
        
            # if image is available
            plt.imsave(dst_fp,img)

print()

# this is to predict the demographics
#-----------------------------------#

count = 1; total = len(df)

for i,v in df.iterrows():
    
    print(f'Predicting demos : {count}/{total} | {i}',
          end='\r');count+=1
    
    dst_fp = dst_dir+i+'/demo.pickle'
    
    if not os.path.exists(dst_fp):
        
        # predict the demographics
        img = plt.imread(dst_dir+i+'/img.jpg')

        # check if image is gscholar icon
        hashed = get_hash(img)

        # if not an icon, detect demographics
        if hashed != 'ffe3c1c3c3e7c3c1':
            demos = get_demo(img)
        else:
            demos = {}

        if demos == {}:
            demos = {
                'angry': np.nan,
                'disgust': np.nan,
                'fear': np.nan,
                'happy': np.nan,
                'sad': np.nan,
                'surprise': np.nan,
                'neutral': np.nan,
                'age': np.nan,
                'gender': np.nan,
                'asian': np.nan,
                'indian': np.nan,
                'black': np.nan,
                'white': np.nan,
                'middle eastern': np.nan,
                'latino hispanic': np.nan}

        with open(dst_fp,'wb') as f:
            pickle.dump(demos,f)
