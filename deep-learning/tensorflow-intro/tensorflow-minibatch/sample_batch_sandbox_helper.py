import os
os.environ['TP_CPP_MIN_LOG_LEVEL']='2'
import math

def batches(batch_size, features, labels):
    assert len(features) == len(labels)
    output_batches = []
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
    return output_batches
