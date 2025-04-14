from types import MethodType

from keras.models import *
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Lambda

from .config import IMAGE_ORDERING
from ..train import train
from ..predict import predict, predict_multiple, evaluate


# source m1 , dest m2
def transfer_weights(m1, m2, verbose=True):

    assert len(m1.layers) == len(
        m2.layers), "Both models should have same number of layers"

    nSet = 0
    nNotSet = 0

    if verbose:
        print("Copying weights ")
        bar = tqdm(zip(m1.layers, m2.layers))
    else:
        bar = zip(m1.layers, m2.layers)

    for l, ll in bar:

        if not any([w.shape != ww.shape for w, ww in zip(list(l.weights),
                                                         list(ll.weights))]):
            if len(list(l.weights)) > 0:
                ll.set_weights(l.get_weights())
                nSet += 1
        else:
            nNotSet += 1

    if verbose:
        print("Copied weights of %d layers and skipped %d layers" %
              (nSet, nNotSet))


#def resize_image(inp,  s, data_format):
#
#    try:
#
#        return Lambda(lambda x: K.resize_images(x,
#                                                height_factor=s[0],
#                                                width_factor=s[1],
#                                                data_format=data_format,
#                                                interpolation='bilinear'))(inp)
#
#    except Exception as e:
#        # if keras is old, then rely on the tf function
#        # Sorry theano/cntk users!!!
#        assert data_format == 'channels_last'
#        assert IMAGE_ORDERING == 'channels_last'
#
#        import tensorflow as tf
#
#        return Lambda(
#            lambda x: tf.image.resize(
#                x, (K.int_shape(x)[1]*s[0], K.int_shape(x)[2]*s[1]))
#        )(inp)

from tensorflow.keras.layers import Lambda
import tensorflow as tf

def resize_image(inp, target_size, data_format):
    """
    Resize the input tensor to the target spatial dimensions.

    Parameters:
    - inp: Input tensor.
    - target_size: Tuple (target_height, target_width) specifying the desired spatial dimensions.
    - data_format: Either 'channels_last' or 'channels_first'.

    Returns:
    - Resized tensor with consistent dimensions.
    """
    def resize_fn(x):
        # Remove any additional dimensions if present
        if len(x.shape) > 4:
            x = tf.squeeze(x, axis=1)  # Squeeze the second dimension if it's 1

        # Perform resizing
        if data_format == "channels_last":
            return tf.image.resize(x, target_size, method="bilinear")
        elif data_format == "channels_first":
            # Transpose to 'channels_last', resize, and transpose back
            x = tf.transpose(x, [0, 2, 3, 1])  # Convert to NHWC
            x = tf.image.resize(x, target_size, method="bilinear")
            return tf.transpose(x, [0, 3, 1, 2])  # Convert back to NCHW
        else:
            raise ValueError("Invalid data_format. Must be 'channels_last' or 'channels_first'.")

    # Determine the output shape dynamically
    if data_format == "channels_last":
        output_shape = (None, target_size[0], target_size[1], inp.shape[-1])
    else:  # 'channels_first'
        output_shape = (None, inp.shape[1], target_size[0], target_size[1])

    return Lambda(resize_fn, output_shape=output_shape, name="resize_image")(inp)

def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height*output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)
    model.evaluate_segmentation = MethodType(evaluate, model)

    return model
