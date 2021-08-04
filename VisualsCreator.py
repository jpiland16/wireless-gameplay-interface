import pickle

from Testing import *
from Visuals import *

FILENAME = "similarity-test.pkl"

# The below (line 9) should be unique, otherwise you will overwrite 
# one of your other sites! Also, be sure that the 'html' folder exists 
# and that the <SITE_NAME> folder exists, inside of the 'html' folder.
SITE_NAME = "similarity-tests"

FOLDER = "html/" + SITE_NAME + "/"

##########################################################################

results = pickle.load(open(FILENAME, "rb"))

generate_site(results, FOLDER)

print("Site created successfully.")