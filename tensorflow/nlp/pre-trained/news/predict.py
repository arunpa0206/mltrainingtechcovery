﻿from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import argparse
from pathlib import Path
import datetime
import os
import random
import string

parser.add_argument('-a', action="store_true", default=False)

parser.add_argument('--output_dir', type=str, help='Where to write the results')

args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

with open(args.text_path,'r') as f:
    text = f.read()

model = BertForSequenceClassification.from_pretrained(args.model_path,num_labels=3,cache_dir=None)

random_filename = ''.join(random.choice(string.ascii_letters) for i in range(10))
output = 'output' + '.csv'
