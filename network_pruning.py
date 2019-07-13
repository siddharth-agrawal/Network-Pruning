import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# batch size for training and inference
batch_size = 128

# network configuration for all layers but the final layer
network_config = [1000, 1000, 500, 200]

# number of epochs to train for
num_epochs = 10

# list of sparsity values to use while pruning
sparsity_list = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]


def get_accuracy(test_images, test_labels, pred_out, input_ph, sess):
    """
    Calculate test accuracy using the ReLU network.

    Arguments:
      :test_images: Array containing all of the images in the test set.
      :test_labels: Labels corresponding to the test images.
      :pred_out: Tensor representing the predictions of the network.
      :input_ph: Placeholder for the input to the network.
      :sess: TF session to run operations.
    Returns:
      :test_accuracy (implicit): Test accuracy with the current state
        of the network.
    """
    num_test_examples = test_images.shape[0]
    num_correct = 0

    # loop over batches for inference
    for batch_idx in range(int(math.ceil(num_test_examples / batch_size))):
        # start and end limits of the batch
        batch_start = batch_idx*batch_size
        batch_end = batch_start + batch_size

        # images and labels for the batch, flatten the images before feeding in
        batch_img = test_images[batch_start:batch_end]
        # reshaping logic to handle the last batch in the test set
        curr_batch_size = batch_img.shape[0]
        batch_img = batch_img.reshape(curr_batch_size, -1)
        batch_labels = test_labels[batch_start:batch_end]

        # get predictions
        predictions = sess.run([pred_out], feed_dict={input_ph: batch_img})
        # keep track of the number of accurate predictions
        num_batch_correct = np.sum(predictions == batch_labels)
        num_correct += num_batch_correct

    return (num_correct / num_test_examples)


def neuron_prune(weights, sparsity):
    """
    Neuron pruning for a weight matrix.

    Arguments:
      :weights: Weight matrix to be pruned.
      :sparsity: Percentage sparsity specified as a fraction.
    Returns:
      :pruned_weights: Pruned weight matrix.
    """
    # cutoff index for the provided sparsity level, it's a
    # fraction of the number of columns for neuron pruning
    cutoff = int(sparsity * weights.shape[1])

    # sorted array of column norms
    weight_col_norms = np.linalg.norm(weights, axis=0)
    sort_col_norms = np.sort(weight_col_norms)

    # cutoff column norm for pruning
    cutoff_norm = sort_col_norms[cutoff]

    # prune away columns whose norms are less than the cutoff,
    # weights are copied so that pruning is not done inplace
    pruned_weights = weights.copy()
    pruned_weights[:, weight_col_norms < cutoff_norm] = 0.0

    return pruned_weights


def neuron_prune_list(weight_list, sparsity):
    """
    Does neuron pruning for a list of weights with the provided level of
    sparsity.

    Arguments:
      :weight_list: List of weights to prune.
      :sparsity: Percentage sparsity specified as a fraction.
    Returns:
      :weight_sparse_list: List of weights after pruning.
    """
    weight_sparse_list = []

    # do neuron pruning for each matrix in the passed list
    for weights in weight_list:
        sparse_weights = neuron_prune(weights, sparsity)
        weight_sparse_list.append(sparse_weights)

    return weight_sparse_list


def weight_prune(weights, sparsity):
    """
    Weight pruning for a weight matrix.

    Arguments:
      :weights: Weight matrix to be pruned.
      :sparsity: Percentage sparsity specified as a fraction.
    Returns:
      :pruned_weights: Pruned weight matrix.
    """
    # cutoff index for the provided sparsity level, it's a
    # fraction of the number of weights for weight pruning
    cutoff = int(sparsity * np.prod(weights.shape))

    # sorted array of absolute weights
    abs_weights = np.abs(weights)
    sort_abs_weights = np.sort(abs_weights, axis=None)

    # cutoff absolute weight for pruning
    cutoff_weight = sort_abs_weights[cutoff]

    # prune away weights whose absolute values are less than
    # the cutoff, weights are copied so that pruning is not
    # done inplace
    pruned_weights = weights.copy()
    pruned_weights[abs_weights < cutoff_weight] = 0.0

    return pruned_weights


def weight_prune_list(weight_list, sparsity):
    """
    Does weight pruning for a list of weights with the provided level of
    sparsity.

    Arguments:
      :weight_list: List of weights to prune.
      :sparsity: Percentage sparsity specified as a fraction.
    Returns:
      :weight_sparse_list: List of weights after pruning.
    """
    weight_sparse_list = []

    # do weight pruning for each matrix in the passed list
    for weights in weight_list:
        sparse_weights = weight_prune(weights, sparsity)
        weight_sparse_list.append(sparse_weights)

    return weight_sparse_list


def get_prune_acc_list(orig_weight_list, weight_upd_ph_list, weight_upd_op_list,
        sess, test_tuple, pred_out, input_ph, prune_method):
    """
    Prunes the weights according to the provided technique and the list of
    sparsity values. Returns a list of test accuracies for each of these
    configurations.

    Arguments:
      :orig_weight_list: List of weight matrices after training.
      :weight_upd_ph_list: List of placeholders for the weight update.
      :weight_upd_op_list: List of weight update operations.
      :sess: TF session to run operations.
      :test_tuple: Tuple containing test images and labels.
      :pred_out: Tensor representing predictions from the network.
      :input_ph: Placeholder for input to the network.
      :prune_method: Either 'weight' or 'neuron' indicating the pruning
        technique to use.
    Returns:
      :accuracy_list: List of test accuracies corresponding to each of the
        sparsity values.
    """
    # extract images and labels from the test data tuple
    test_images, test_labels = test_tuple

    accuracy_list = []
    # for each value of sparsity, prune the weights and record test accuracy
    for sparsity in sparsity_list:
        # get pruned weights
        if prune_method == 'weight':
            sparse_weight_list = weight_prune_list(orig_weight_list, sparsity)
        if prune_method == 'neuron':
            sparse_weight_list = neuron_prune_list(orig_weight_list, sparsity)

        feed_dict_sparse = {}
        # create the fedd dictionary for the weight update
        for sparse_weights_i, weight_upd_ph_i in \
                zip(sparse_weight_list, weight_upd_ph_list):
            feed_dict_sparse[weight_upd_ph_i] = sparse_weights_i
        # update the network with the pruned weights
        sess.run(weight_upd_op_list, feed_dict=feed_dict_sparse)

        # compute test accuracy with the updated weights
        accuracy_list.append(get_accuracy(test_images, test_labels, pred_out,
            input_ph, sess))

    return accuracy_list


def weight_assign(layer_weight_list):
    """
    Function that creates assignment operations for a list of weight
    tensors using shape-equivalent placeholders.

    Arguments:
      :layer_weight_list: List of weight tensors to create assignment
        operations for.
    Returns:
      :ph_list: List of placeholders to feed in the values for assignment.
      :op_list: List of the tensor assignment operations.
    """
    ph_list = []
    op_list = []

    # for each weight tensor, create an assingment operation
    # using a placeholder
    for weights_i in layer_weight_list:
        weights_i_ph = tf.placeholder(tf.float32,
            weights_i.get_shape().as_list())
        ph_list.append(weights_i_ph)
        op_list.append(tf.assign(weights_i, weights_i_ph))

    return ph_list, op_list


def relu_network(batch_input, batch_label, num_classes, dataset):
    """
    Function that constructs the network graph and loss function.

    Arugments:
      :batch_input: Input tensor for the network.
      :batch_label: Tensor for the ground truth labels.
      :num_classes: Number of classes for the classification task.
      :dataset: Either 'mnist' or 'fashion_mnist' indicating the
        dataset being used.
    Returns:
      :pred_out: Tensor for the predictions of the network.
      :ce_loss: Tensor with the loss value for the network.
    """
    # get the pixel values in the [0, 1] range before feeding in
    network_out = batch_input / 255

    # define relu network based on the provided config
    for layer_idx, num_units in enumerate(network_config):
        # layers do not have bias components and are named for
        # identification while pruning
        network_out = tf.layers.dense(network_out, num_units,
            activation=tf.nn.relu, use_bias=False,
            name='{}/layer_{}'.format(dataset, layer_idx))

    # final layer for classification
    network_out = tf.layers.dense(network_out, num_classes)

    # predict class with the maximum probability, argmax can be done
    # before softmax
    pred_out = tf.argmax(network_out, axis=-1)

    # cross entropy loss for classification
    ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(batch_label, tf.int32), logits=network_out))

    return pred_out, ce_loss


def network_pruning(dataset):
    """
    The central function that does all the work for a dataset. This includes
    loading data, creating and training the network, pruning using the two
    different techniques and then plotting the results.

    Arguments:
      :dataset: Either 'mnist' or 'fashion_mnist' indicating the dataset
        to work on.
    Returns:
      None
    """
    # load the relevant dataset available in tf, automatically downloads if
    # not available locally
    if dataset == 'mnist':
        train_tuple, test_tuple = tf.keras.datasets.mnist.load_data()
    if dataset == 'fashion_mnist':
        train_tuple, test_tuple = tf.keras.datasets.fashion_mnist.load_data()

    # images and corresponding labels for training
    train_images, train_labels = train_tuple
    # image array has dimensions [num_images, height, width]
    num_train_examples, img_height, img_width = train_images.shape
    # number of classes, add 1 to account for zero-indexing
    num_classes = np.max(train_labels) + 1

    # placeholder for network input and the corresponding labels,
    # image is flattened before feeding in
    input_ph = tf.placeholder(tf.uint8, [None, img_height*img_width])
    label_ph = tf.placeholder(tf.uint8, [None])

    # get tensors for network predictions and the cross entropy loss
    pred_out, ce_loss = relu_network(input_ph, label_ph, num_classes, dataset)

    # training operation for the network
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(ce_loss)

    # get session to run operations and intialize variables
    init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())
    sess = tf.Session(); sess.run(init_op)

    # train the network
    for step in range((num_epochs * num_train_examples) // batch_size):
        # select random examples from the training set
        rand_batch = np.random.choice(num_train_examples, batch_size)
        # flatten the images before passing them
        batch_img = train_images[rand_batch].reshape(batch_size, -1)
        batch_label = train_labels[rand_batch]

        # training step
        sess.run(train_op, feed_dict={input_ph: batch_img,
            label_ph: batch_label})

    # collect all weight tensors for the relu layers
    layer_weight_list = [weights_i for weights_i in tf.trainable_variables()
        if '{}/layer_'.format(dataset) in weights_i.name]

    # get list of placeholders and assignment operations to update weights
    # after pruning
    weight_upd_ph_list, weight_upd_op_list = weight_assign(layer_weight_list)

    # get value of weight matrices after training
    orig_weight_list = sess.run(layer_weight_list)

    # get test accuracies with weight pruning at various sparsities
    weight_prune_acc = get_prune_acc_list(orig_weight_list,
        weight_upd_ph_list, weight_upd_op_list, sess, test_tuple,
        pred_out, input_ph, 'weight')

    # get test accuracies with neuron pruning at various sparsities
    neuron_prune_acc = get_prune_acc_list(orig_weight_list,
        weight_upd_ph_list, weight_upd_op_list, sess, test_tuple,
        pred_out, input_ph, 'neuron')

    # plot the results of pruning
    plt.plot(sparsity_list, weight_prune_acc, '--o', label='Weight pruning')
    plt.plot(sparsity_list, neuron_prune_acc, '--o', label='Neuron pruning')
    plt.xlabel('Sparsity'); plt.ylabel('Test Accuracy')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(dataset))
    plt.clf()


if __name__ == '__main__':
    # network pruning for mnist
    network_pruning('mnist')
    # network pruning for fashion-mnist
    network_pruning('fashion_mnist')
