from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def define_rnn_layer(hidden_units, cell_type='GRU', bidirect_flag=False,
                     rnn_U_drop=0.0, rnn_W_drop=0.0, layer_name='rnn'):
    
    if cell_type == 'Simple':
        rnn_layer = SimpleRNN(n_units, activation='relu',
                              return_sequences=True, implementation=2, 
                              recurrent_dropout=rnn_U_drop, dropout=rnn_W_drop,
                              name=layer_name)
        if bidirect_flag:
            return Bidirectional(rnn_layer)
        else:
            return rnn_layer
    elif cell_type == 'GRU':
        rnn_layer = GRU(hidden_units, 
                        return_sequences=True, implementation=2, 
                        recurrent_dropout=rnn_U_drop, dropout=rnn_W_drop,
                        name=layer_name)
        if bidirect_flag:
            return Bidirectional(rnn_layer)
        else:
            return rnn_layer
    elif cell_type == 'LSTM':
        rnn_layer = LSTM(hidden_units, 
                         return_sequences=True, implementation=2, 
                         recurrent_dropout=rnn_U_drop, dropout=rnn_W_drop, 
                         name=layer_name)
        if bidirect_flag:
            return Bidirectional(rnn_layer)
        else:
            return rnn_layer
    else:
        raise ValueError('"rnn_cell" can only take on any of {}, {}, or {} values'.format(['GRU', 'LSTM', 'Simple']))

def asr_model(input_dim, filters, kernel_size, conv_stride, conv_dilate,
                conv_border_mode, units, output_dim=29, 
                n_rnn_layers=1, rnn_cell='GRU', rnn_bidirect=False,
                cnn_out_drop=0.0, rnn_U_drop=0.0, rnn_W_drop=0.0, rnn_out_drop=0.0):
                
    """ Build a deep, bidirectional recurrent + convolutional network for speech 
        
        rnn_cell: ['GRU', 'LSTM', 'Simple']
    """    
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d) 
    
    bn_cnn = Activation('relu', name='bn_cnn')(bn_cnn)
    
    if cnn_out_drop > 0.0:
        bn_cnn = Dropout(cnn_out_drop, name='dropout_bn_cnn')(bn_cnn)
    
    # create first rnn layer
    rnn_layer_1 = define_rnn_layer(hidden_units=units, cell_type=rnn_cell, 
                                   rnn_U_drop=rnn_U_drop, rnn_W_drop=rnn_W_drop,
                                   bidirect_flag=rnn_bidirect,
                                   layer_name='rnn_1')
    rnn_1 = rnn_layer_1(bn_cnn)
    # create first bathnorm layer   
    bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(rnn_1)          

    # create subsequent rnn and batchnorm layers 
    previous_bn_rnn_out = bn_rnn_1
    for layer_id in range(1, n_rnn_layers):
        layer_name = "rnn_" +  str(layer_id+1)
        rnn_layer = define_rnn_layer(hidden_units=units, cell_type=rnn_cell, 
                                     rnn_U_drop=rnn_U_drop, rnn_W_drop=rnn_W_drop,
                                     bidirect_flag=rnn_bidirect,
                                     layer_name=layer_name)           
        rnn = rnn_layer(previous_bn_rnn_out)
        
        batchnorm_name = "bn_rnn_" + str(layer_id+1)
        next_bn_rnn_out = BatchNormalization(name=batchnorm_name)(rnn)        
        
        previous_bn_rnn_out = next_bn_rnn_out   
        
    if rnn_out_drop > 0.0:
        previous_bn_rnn_out = Dropout(rnn_out_drop, name='dropout_rnn_out')(previous_bn_rnn_out)                      

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='dense')(previous_bn_rnn_out)  
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation=conv_dilate)
    print(model.summary())
    return model        