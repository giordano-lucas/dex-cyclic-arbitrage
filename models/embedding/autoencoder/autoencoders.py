from tensorflow import keras
from tensorflow.keras import layers

def linear():
    model_name = "linear"
    in_shape = (3,600, 2) 
    # build encoder
    input_layer = keras.Input(shape=in_shape)
    x = layers.Reshape([in_shape[0]*in_shape[1]*in_shape[2]])(input_layer)
    x = layers.Dense(1800,  activation='linear')(x)
    x = layers.Dense(100,  activation='linear')(x)
    x = layers.Dense(1800,  activation='linear')(x)
    x = layers.Dense(in_shape[0]*in_shape[1]*in_shape[2],  activation='linear')(x)
    output_layer =layers.Reshape(in_shape)(x)
    
    
    # combine encoder and decoder
    autoencoder = keras.Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',)
    return model_name ,autoencoder

def fully_connected_3L():
    model_name = "fully_connected_3L"
    in_shape = (3,600, 2) 
    # build encoder
    input_layer = keras.Input(shape=in_shape)
    x = layers.Reshape([in_shape[0]*in_shape[1]*in_shape[2]])(input_layer)
    x = layers.Dense(600,  activation='elu')(x)
    x = layers.Dense(100,  activation='elu')(x)
    x = layers.Dense(600,  activation='elu')(x)
    x = layers.Dense(in_shape[0]*in_shape[1]*in_shape[2],  activation='elu')(x)
    output_layer =layers.Reshape(in_shape)(x)
    
    
    # combine encoder and decoder
    autoencoder = keras.Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',)
    return model_name ,autoencoder

def CNN():
    model_name = "CNN"
    # build encoder
    input_img = keras.Input(shape=(3,600, 2))
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(1, (2, 2), activation='relu', padding='same')(x)
    # build decoder
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((1, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((3, 3))(x)
    decoded = layers.Conv2D(2, (3, 3), activation='relu', padding='same')(x)
    # combine encoder and decoder
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',)
    return model_name, autoencoder

def CNN_fully_connected():
    model_name = "CNN_fully_connected"
    # build encoder
    input_img = keras.Input(shape=(3,600, 2))
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((3, 1), padding='same')(x)
    encoded = layers.Conv2D(1, (1, 2), activation='relu', padding='same')(x)
    # Dense layers
    x = layers.Dense(300,  activation='elu')(x)
    encoded = layers.Dense(100,  activation='elu')(x)
    x = layers.Dense(100,  activation='elu')(encoded)
    x = layers.Dense(300,  activation='elu')(x)
    # build decoder
    x = layers.Conv2D(4, (1, 2), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((3, 1))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((1, 2))(x)
    decoded = layers.Conv2D(2, (3, 3), activation='relu', padding='same')(x)
    # combine encoder and decoder
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',)
    return model_name, autoencoder

def talos_architecture(params):
    # unique name
    model_name = 'talos_' + "_".join([str(v) for v in params.values()])
    in_shape = (3,600, 2) 
    # build encoder
    input_layer = keras.Input(shape=in_shape)
    x = layers.Reshape([in_shape[0]*in_shape[1]*in_shape[2]])(input_layer)
    for l in range(params['dense_layers']):
        x = layers.Dropout(params['dropout'])(x)
        if l == 0 or l == params['dense_layers'] - 1:
            x = layers.Dense(params['first_neuron'],  activation=params["activation"])(x)
        else:
            x = layers.Dense(100,  activation=params["activation"])(x)
    x = layers.Dense(in_shape[0]*in_shape[1]*in_shape[2],  activation=params["activation"])(x)
    output_layer =layers.Reshape(in_shape)(x)

    # combine encoder and decoder
    autoencoder = keras.Model(input_layer, output_layer)
    autoencoder.compile(optimizer=params['optimizer'], loss='mean_squared_error',)
<<<<<<< HEAD
    return model_name ,autoencoder
=======
    return model_name ,autoencoder
>>>>>>> 2b113825b93622f4f7acce54610bc578cff984a4
