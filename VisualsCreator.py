import pickle, os

from Testing import *
from Visuals import *

FILENAME = "similarity-test.pkl"

# The below (line 9) should be unique, otherwise you will overwrite 
# one of your other sites! 
SITE_NAME = "similarity-tests"

FOLDER = "html/" + SITE_NAME + "/"

##########################################################################

if not os.path.exists('html'):
    os.mkdir('html')

if not os.path.exists('html/' + SITE_NAME):
    os.mkdir('html/' + SITE_NAME)

results = pickle.load(open(FILENAME, "rb"))

generate_site(results, FOLDER)

print("Site created successfully.")