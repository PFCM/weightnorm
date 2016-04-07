"""Evaluate both weight-normalised and normal feed forward networks on MNIST."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import mnist
import input_data

VALID_PERF = []

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_string('data_dir', 'data', 'Directory for storing data')
flags.DEFINE_string('logdir', 'logs', 'Where to write summaries')
flags.DEFINE_boolean('weightnorm', False, 'whether to normalise weights')
flags.DEFINE_integer('batch_size', 100, 'size of batch')
flags.DEFINE_integer('hidden_size', 256, 'size of hidden layer')
flags.DEFINE_integer('num_layers', 1, 'number of hidden layers')
flags.DEFINE_float('momentum', 0.9, 'momentum for the optimiser')
flags.DEFINE_boolean('batchnorm', False, 'whether to try and use batchnorm or not')


def fill_feed(dataset, inputs, labels, batch_size):
    """Constructs and returns a new feed dict"""
    image_data, label_data = dataset.next_batch(batch_size)
    return {
        inputs: image_data,
        labels: label_data
        }


def evaluate(sess,
             data,
             logits,
             eval_op,
             batch_size,
             input_var,
             label_var,
             global_step,
             writer):
    """Evaluates the ability of the logits to predict the given dataset.
    Optionally runs the summary op and returns the result"""
    total_correct = 0
    total_size = 0
    num_batches = data.num_examples // batch_size
    summaries = tf.merge_all_summaries()
    #print('\n..evaluating', end='')
    for i in range(num_batches):
        data_dict = fill_feed(data, input_var, label_var, batch_size)
        num_correct, cost = sess.run([eval_op[0], eval_op[1]],
                                     feed_dict=data_dict)
        total_correct += num_correct
        total_size += batch_size
        #writer.add_summary(summ_str, global_step.eval(session=sess))
        if i % 10 == 0:
            print('\r..evaluating: {}/{} ({:.4f})      '.format(total_correct, total_size,
                                                      total_correct/total_size),
                  end='')
    return total_correct / total_size
        

def main(_):
    # build a model
    data = input_data.read_data_sets(FLAGS.data_dir, one_hot=False,
                                     fake_data=False)
    # make placeholders
    images = tf.placeholder(tf.float32, [FLAGS.batch_size, mnist.IMAGE_PIXELS],
                            name='inputs')
    labels = tf.placeholder(tf.int32, [FLAGS.batch_size], name='labels')

    # build model up to inference
    logits = mnist.inference(images, FLAGS.hidden_size, FLAGS.num_layers,
                             do_weightnorm=FLAGS.weightnorm,
                             do_batchnorm=FLAGS.batchnorm,
                             train=True)
    if not FLAGS.batchnorm:
        eval_logits = logits
    else:
        eval_logits = mnist.inference(images, FLAGS.hidden_size, FLAGS.num_layers,
                                      do_weightnorm=FLAGS.weightnorm,
                                      do_batchnorm=FLAGS.batchnorm,
                                      train=False)

    # get a loss function
    loss = mnist.loss(logits, labels, 'train_xent')
    eval_loss = mnist.loss(eval_logits, labels, 'eval_xent')
    # add a sumary of this to track the training loss
   
    # get training ops
    train_op, gstep = mnist.training(loss, FLAGS.learning_rate, FLAGS.momentum)

    # get an op to return precision on a batch
    eval_op = mnist.evaluation(eval_logits, labels)

    valid_var = tf.Variable(0, 'validation performance')
    valid_summ = tf.scalar_summary('validation accuracy', valid_var)

    # get summary op
    summarise = tf.merge_all_summaries()
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
        tf.initialize_all_variables().run()
        # do some training
        print('nb: {} steps per epoch'.format(
            data.train.num_examples // FLAGS.batch_size))
        print('Step 0/{}.'.format(FLAGS.max_steps), end='')
        for i in range(FLAGS.max_steps):
            if (i+1) % 5 == 0:
                # write summaries, check on validation set
                if (i+1) % 100 == 0:
                    valid_perf = evaluate(sess,
                                          data.validation,
                                          logits,
                                          [eval_op, eval_loss],
                                          FLAGS.batch_size,
                                          images,
                                          labels,
                                          gstep,
                                          writer)
                    print()
                summ_str, _, _ = sess.run([summarise, loss, train_op],
                                          fill_feed(data.train, images, labels,
                                                    FLAGS.batch_size))
                writer.add_summary(summ_str, gstep.eval(session=sess))
            else:
                # do a step of training
                loss_val, _ = sess.run([loss, train_op],
                                   fill_feed(data.train, images, labels,
                                             FLAGS.batch_size))
                print('\rStep {} (loss {})'.format(i+1, loss_val), end='', flush=True)
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Test evaluation:')
        evaluate(sess, data.test, logits, [eval_op, eval_loss], FLAGS.batch_size, images, labels, gstep, writer)
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    

if __name__ == '__main__':
    tf.app.run()
