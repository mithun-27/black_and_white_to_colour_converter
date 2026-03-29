from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def build_transfer_unet(input_shape=(256, 256, 3)):
    """
    Builds a Transfer Learning U-Net using MobileNetV2 as the encoder.
    The input is expected to be a 3-channel (repeated) L channel.
    """
    # 1. Base Encoder (Pre-trained on ImageNet)
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # 2. Extract Skip Connections
    # We choose specific layers for skip connections to preserve details
    layer_names = [
        'block_1_expand_relu',   # 128x128
        'block_3_expand_relu',   # 64x64
        'block_6_expand_relu',   # 32x32
        'block_13_expand_relu',  # 16x16
    ]
    skip_layers = [base_model.get_layer(name).output for name in layer_names]
    
    # 3. Bottleneck (the deepest part of the encoder)
    bottleneck = base_model.output # 8x8x1280
    
    # 4. Decoder with Skip Connections
    x = bottleneck
    
    # Upsample to 16x16
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip_layers[3]])
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Upsample to 32x32
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip_layers[2]])
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Upsample to 64x64
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip_layers[1]])
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Upsample to 128x128
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip_layers[0]])
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Upsample to 256x256
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    
    # 5. Output Layer
    # Predict 2-channel AB normalized to [-1, 1]
    outputs = Conv2D(2, (1, 1), activation='tanh')(x)
    
    # 6. Final Model
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Freeze the base encoder for initial training
    base_model.trainable = False
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_transfer_unet()
    model.summary()
    print("\nTransfer Learning Model built successfully!")
