import tensorflow as tf

def Tanimoto_loss(label, pred):
    """
    Implementation of Tanimoto loss in tensorflow 2.x
    -------------------------------------------------------------------------
    Tanimoto coefficient with dual from: Diakogiannis et al 2019 (https://arxiv.org/abs/1904.00592)
    """
    smooth = 1e-5

    Vli = tf.math.reduce_mean(tf.math.reduce_sum(label, axis=[1,2]), axis=0)
    # wli =  1.0/Vli**2 # weighting scheme
    wli = tf.math.reciprocal(Vli**2) # weighting scheme

    # ---------------------This line is taken from niftyNet package --------------
    # ref: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py, lines:170 -- 172
    # First turn inf elements to zero, then replace that with the maximum weight value
    new_weights = tf.where(tf.math.is_inf(wli), tf.zeros_like(wli), wli)
    wli = tf.where(tf.math.is_inf(wli), tf.ones_like(wli) * tf.math.reduce_max(new_weights), wli)
    # --------------------------------------------------------------------
    square_pred = tf.math.square(pred)
    square_label = tf.math.square(label)
    add_squared_label_pred = tf.math.add(square_pred, square_label)
    sum_square = tf.math.reduce_sum(add_squared_label_pred, axis=[1, 2])
    product = tf.math.multiply(pred, label)
    sum_product = tf.math.reduce_sum(product, axis=[1, 2])
    sum_product_labels = tf.math.reduce_sum(tf.math.multiply(wli, sum_product), axis=-1)
    denomintor = tf.math.subtract(sum_square, sum_product)

    denomintor_sum_labels = tf.math.reduce_sum(tf.math.multiply(wli, denomintor), axis=-1)
    loss = tf.math.divide(sum_product_labels + smooth, denomintor_sum_labels + smooth)
    return loss


def Tanimoto_dual_loss():
    '''
        Implementation of Tanimoto dual loss in tensorflow 2.x
        ------------------------------------------------------------------------
            Note: to use it in deep learning training use: return 1. - 0.5*(loss1+loss2)
            OBS: Do use note's advice. Otherwise tanimoto doesn't work
    '''
    def loss(label, pred):
        loss1 = Tanimoto_loss(pred, label)
        pred = tf.math.subtract(1.0, pred)
        label = tf.math.subtract(1.0, label)
        loss2 = Tanimoto_loss(label, pred)
        loss = (loss1+loss2)*0.5
        return 1.0 - loss
    return loss
