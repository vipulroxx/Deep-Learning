import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sample_batch_sandbox_helper import batches
from pprint import pprint
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]
pprint(batches(3, example_features, example_labels))
