#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a custom Estimator for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import iris_data_tpu as iris_data

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url. To find out the name or ip address of TPU, use command "
    "'gcloud compute tpus list --zone=<zone-name>'")

# Model specific parameters
tf.flags.DEFINE_string("model_dir",
        default=None,
        help="Estimator model_dir")
tf.flags.DEFINE_integer("batch_size",
        default=128,
        help="This is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps",
        default=1000,
        help="Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps",
        default=4,
        help="Total number of evaluation steps. If `0`, evaluation "
        "after training is skipped.")

tf.flags.DEFINE_bool("use_tpu",
        default=True,
        help="Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations",
        default=500,
        help="Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards",
        default=8,
        help="Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS

def metric_fn(labels, logits):
    """Function to return metrics for evaluation"""

    predicted_classes = tf.argmax(logits, 1)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    return {'accuracy': accuracy}
    

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))

    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    tf.enable_eager_execution()

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Resolve TPU cluster and runconfig for this.
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu)

    run_config = tf.contrib.tpu.RunConfig(
            model_dir=FLAGS.model_dir,
            cluster=tpu_cluster_resolver,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations),
            )

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.contrib.tpu.TPUEstimator(
        model_fn=my_model,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size,
        config=run_config,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
            'use_tpu': FLAGS.use_tpu,
        })

    # Train the Model.
    classifier.train(
            input_fn = lambda params: iris_data.train_input_fn(
                train_x, train_y, params["batch_size"]),
            max_steps=FLAGS.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn = lambda params: iris_data.eval_input_fn(
            test_x, test_y, params["batch_size"]),
        steps=FLAGS.eval_steps)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    predictions = classifier.predict(
        input_fn = lambda params: iris_data.predict_input_fn(
            iris_data.PREDICTION_INPUT_DATA, params["batch_size"]))
    
    for pred_dict, expec in zip(predictions, iris_data.PREDICTION_OUTPUT_DATA):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
