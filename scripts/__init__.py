#!/usr/bin/python
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, redirect, url_for, session, request, render_template
from werkzeug.utils import secure_filename
import os
import json
import time
import numpy as np
import tensorflow as tf
from .label_image import read_tensor_from_image_file, load_graph, load_labels

UPLOAD_FOLDER = '/uploads'

app = Flask(__name__)
# sslify = SSLify(app)
app.debug = True
app.secret_key = 'development'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send', methods=['POST'])
def send():
  image = request.files['image']
  filename = secure_filename(image.filename)
  image.save(os.path.join('./uploads', filename))

  file_name = os.path.join('./uploads',filename)
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Mul"
  output_layer = "final_result"

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)
  results = np.array(json.loads(json.dumps(results.tolist())))

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
  template = "{} (score={:0.5f})"
  analysis = {}
  for i in top_k:
    print(template.format(labels[i], results[i]))
    analysis[labels[i]] = results[i]
  return json.dumps(analysis)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

