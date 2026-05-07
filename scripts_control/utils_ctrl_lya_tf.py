import tensorflow as tf


class ControllerTF(tf.keras.Model):
    """CNN-based vision controller with action clamping."""

    def __init__(self, img_size=64):
        super(ControllerTF, self).__init__()

        bn_momentum = 0.9
        bn_epsilon = 1e-5

        self.backbone = tf.keras.Sequential([

            # 0
            tf.keras.layers.Conv2D(
                16,
                kernel_size=5,
                strides=2,
                padding='same',
                use_bias=True,
                name='backbone.0'
            ),

            # 1
            tf.keras.layers.BatchNormalization(
                momentum=bn_momentum,
                epsilon=bn_epsilon,
                name='backbone.1'
            ),

            # 2
            tf.keras.layers.ReLU(
                name='backbone.2'
            ),

            # 3
            tf.keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=2,
                padding='same',
                use_bias=True,
                name='backbone.3'
            ),

            # 4
            tf.keras.layers.BatchNormalization(
                momentum=bn_momentum,
                epsilon=bn_epsilon,
                name='backbone.4'
            ),

            # 5
            tf.keras.layers.ReLU(
                name='backbone.5'
            ),

            # 6
            tf.keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=2,
                padding='same',
                use_bias=True,
                name='backbone.6'
            ),

            # 7
            tf.keras.layers.BatchNormalization(
                momentum=bn_momentum,
                epsilon=bn_epsilon,
                name='backbone.7'
            ),

            # 8
            tf.keras.layers.ReLU(
                name='backbone.8'
            ),

            # 9
            tf.keras.layers.GlobalAveragePooling2D(
                name='backbone.9'
            ),

            # 10
            tf.keras.layers.Flatten(
                name='backbone.10'
            ),
        ])

        self.action_head = tf.keras.Sequential([

            # 0
            tf.keras.layers.Dense(
                64,
                use_bias=True,
                name='action_head.0'
            ),

            # 1
            tf.keras.layers.ReLU(
                name='action_head.1'
            ),

            # 2
            tf.keras.layers.Dropout(
                0.1,
                name='action_head.2'
            ),

            # 3
            tf.keras.layers.Dense(
                6,
                use_bias=True,
                name='action_head.3'
            ),
        ])

        self.scale_t = 1.0
        self.scale_r = 0.3

    def call(self, x, training=False):

        features = self.backbone(
            x,
            training=training
        )

        action = self.action_head(
            features,
            training=training
        )

        v_t = action[:, :3] * self.scale_t
        v_r = action[:, 3:] * self.scale_r

        v_t = tf.clip_by_value(v_t, -1.0, 1.0)
        v_r = tf.clip_by_value(v_r, -0.3, 0.3)

        return tf.concat([v_t, v_r], axis=-1)

class LyapunovTF(tf.keras.Model):

    def __init__(
        self,
        hidden_dims=None,
        pos_scale=2.0,
        ang_scale=3.0
    ):

        super(LyapunovTF, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 32]

        self.alpha_net = tf.keras.Sequential([

            # 0
            tf.keras.layers.Dense(
                hidden_dims[0],
                activation=None,
                use_bias=True,
                name='alpha_net.0'
            ),

            # 1
            tf.keras.layers.ReLU(
                name='alpha_net.1'
            ),

            # 2
            tf.keras.layers.Dense(
                hidden_dims[1],
                activation=None,
                use_bias=True,
                name='alpha_net.2'
            ),

            # 3
            tf.keras.layers.ReLU(
                name='alpha_net.3'
            ),

            # 4
            tf.keras.layers.Dense(
                1,
                activation=None,
                use_bias=True,
                name='alpha_net.4'
            ),
        ])

        self.pos_scale = pos_scale
        self.ang_scale = ang_scale
        self.v_scale = 3.0

    def call(self, x, target, training=False):

        pos_error = x[:, :3] - target[:, :3]

        ang_error = x[:, 3:] - target[:, 3:]

        pos_term = tf.reduce_sum(
            tf.square(pos_error),
            axis=-1
        )

        ang_term = tf.reduce_sum(
            1.0 - tf.cos(ang_error),
            axis=-1
        )

        pos_norm = tf.norm(
            pos_error,
            axis=-1,
            keepdims=True
        )

        ang_norm = tf.expand_dims(
            ang_term,
            axis=-1
        )

        pos_norm_scaled = (
            pos_norm / self.pos_scale
        )

        ang_norm_scaled = (
            ang_norm / self.ang_scale
        )

        alpha_input = tf.concat(
            [pos_norm_scaled, ang_norm_scaled],
            axis=-1
        )

        alpha_logit = self.alpha_net(
            alpha_input,
            training=training
        )

        alpha = tf.sigmoid(
            alpha_logit
        )[:, 0]

        V = self.v_scale * (
            alpha * pos_term
            + (1.0 - alpha) * ang_term
        )

        eps = 1e-6

        alpha_clamped = tf.clip_by_value(
            alpha,
            eps,
            1 - eps
        )

        alpha_reg = -tf.reduce_mean(
            tf.math.log(
                4.0
                * alpha_clamped
                * (1 - alpha_clamped)
            )
        )

        return V, alpha_reg