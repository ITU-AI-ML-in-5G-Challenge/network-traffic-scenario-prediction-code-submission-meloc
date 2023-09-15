import torch    

'''
### Encoder and Decoder Blocks for UNET model
'''

class encoder_block (torch.nn.Module):
    
    '''
    Encoder_Block Arguments:
        
        -input_channels:    Input Channels of the block
        
        -output_channels:   The first CNN-Layer takes the input from x channels to
                            y channels where x = input_channels and y = output_channels.
                            Each consecutive layer keeps the number of channels constant.
        
        -activation_func:   The activation function used after each CNN-layer
        
        -kernel_size:       Kernel_size of the CNN-layers.
        
        -no_layers:         Total number of CNN-layers.
        
        -pooling:           Type of final pooling layer. If pooling == 'avg_pooling' use
                            average pooling, else use maximum pooling.
        
        -pooling_stride:    Stride of the final pooling layer (determines by how much the 
                            length of the input tensor will be reduced by running through 
                            the encoder block)
        
        -use_norm:          Boolean; If True: Use Batchnorm after the last CNN-Layer and 
                            before the final activation.
                            
                            NOTE: My current understandin is that Batchnorm should usually be
                            applied before the CNN-Layer and not after. However some
                            (small) experiments with earlier versions of the final models
                            showed better peak performance when using the batchnorm after
                            the CNN-Layer.
    '''
    
    def __init__(self, input_channels, output_channels, activation_func, kernel_size, no_layers, pooling, pooling_stride, use_norm):
        
        super().__init__()
        
        self.no_layers = no_layers
        self.use_norm = use_norm
        self.channels = [input_channels] + no_layers*[output_channels]
        self.cnn_list = torch.nn.ModuleList([torch.nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size = kernel_size, padding = int(kernel_size/2), padding_mode = 'zeros') for i in range(no_layers)])
        
        if use_norm:
            self.norm = torch.nn.BatchNorm1d(output_channels)
        
        self.act_func = activation_func
        
        if pooling == 'avg_pool':
            self.pooling_layer = torch.nn.AvgPool1d(kernel_size = pooling_stride, stride = pooling_stride)
        else:
            self.pooling_layer = torch.nn.MaxPool1d(kernel_size = pooling_stride, stride = pooling_stride)
          
            
    def forward(self, x):
        
        for i in range(self.no_layers):
            x = self.cnn_list[i](x)
            
            if self.use_norm and i == (self.no_layers - 1):
                x = self.norm(x)
            
            x = self.act_func(x)
        
        return (x, self.pooling_layer(x))



class decoder_block (torch.nn.Module):
    
    '''
    Decoder_Block Arguments:
        
        -up_input_channels:     Number of input channels for the upsampling layer
        
        -up_output_channels:    Number of output channels for the upsampling layer
                                (only relevant if using ConvTranspose1d Layer)  
                                
        -conv_input_channels:   Input Channels for the first CNN-Layer after upsampling.
        
        -output_channels:       The first CNN-Layer takes the input from x channels to
                                y channels where x = conv_input_channels and y = output_channels.
                                Each consecutive layer keeps the number of channels constant.  
        
        -activation_func:       The activation function used after each CNN-layer
        
        -kernel_size:           Kernel size of the CNN-layers.
        
        -no_layers:             Number of CNN-layers. 
        
        -upconv_type:           String; if upconv_type == 'upsample' use a simple linear
                                Upsample-Layer for upsampling. Else use a transposed convolution.
        
        -upconv_stride:         When using transposed convolution, this is the stride and kernel_size of
                                of the convolution, when using an Upsample-Layer this is the scaling factor.
                                In both cases the length of the input tensor is multiplied by upconv_stride
                                afterwards.
        
        -use_norm:              Boolean; If True: Use Batchnorm after the last CNN-Layer and 
                                before the final activation.
                                
                                NOTE: My current understandin is that Batchnorm should usually be
                                applied before the CNN-Layer and not after. However some
                                (small) experiments with earlier versions of the final models
                                showed better peak performance when using the batchnorm after
                                the CNN-Layer.
                                
    '''
    
    def __init__(self, up_input_channels, up_output_channels, conv_input_channels, output_channels, activation_func, kernel_size, no_layers, upconv_type, upconv_stride, use_norm):
        
        super().__init__()
        
        self.no_layers = no_layers
        self.use_norm = use_norm
        
        if upconv_type == 'upsample':
            self.up_layer = torch.nn.Upsample(scale_factor = upconv_stride, mode = 'linear')
        else:
            self.up_layer = torch.nn.ConvTranspose1d(up_input_channels, up_input_channels, kernel_size = upconv_stride, stride = upconv_stride)

        self.channels = [conv_input_channels] + no_layers*[output_channels]
        self.cnn_list = torch.nn.ModuleList([torch.nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size = kernel_size, padding = int(kernel_size/2), padding_mode = 'zeros') for i in range(no_layers)])
        
        if use_norm:
            self.norm = torch.nn.BatchNorm1d(output_channels)
        
        self.act_func = activation_func
        
        
    def forward(self, x, cat_features = False, y = None ):
        
        x = self.up_layer(x)
        
        if cat_features == True:
            x = torch.cat([x,y], dim = 1)
        
        for i in range(self.no_layers):
            x = self.cnn_list[i](x)
            
            if self.use_norm and i == (self.no_layers - 1):
                x = self.norm(x)
            
            x = self.act_func(x)
            
        return x
    

class unet_model (torch.nn.Module):
    
    '''
    UNET_Model is initialized with a dictionary "params". The dictionary needs to have
    the following keys:
        
        'input_features','no_blocks', 'cat_features', 'use_skip',
        'enc_channels', 'enc_activation','enc_kernel_size', 'enc_no_layers', 
        'enc_pooling', 'enc_pooling_stride','enc_use_norm',
        'trans_layers','trans_channels','trans_kernel_size',
        'trans_activation',
        'dec_channels', 'dec_up_output_channels', 'dec_conv_input_channels',
        'dec_activation_func', 'dec_kernel_size', 'dec_no_layers', 'dec_upconv_type', 
        'dec_upconv_stride', 'dec_use_norm','output_channels', 'output_func',
        'use_separate_input', 'si_input_channels', 'si_output_channels'
    
    The values of these dictionary entries have the following meaning as input
    arguments for the unet_model:
        
        -enc_*:             These are the input arguments for the encoder blocks.
                            channels, kernel_size and pooling_stride are handed to the
                            model object as lists, where each list element determines the
                            output_channels, kernel_size and pooling_stride of each 
                            consecutive block respectively. activation, pooling, no_layers
                            and use_norm are handed to the model as single variables and
                            the same value is used for all the encoder blocks of the model
                        
        -dec_*:             These are the input arguments for the decoder blocks.
                            channels, up_output_channels, conv_input_chanmels, kernel_size
                            and upconv_stride are handed to the model object as lists, 
                            where each list element determines the channels, 
                            up_output_channels, conv_input_chanmels, kernel_size
                            and upconv_stride of each consecutive block respectively. 
                            activation, no_layers, upconv_type and use_norm are handed to 
                            the model as single variables and the same value is used for 
                            all the decoder blocks of the model.
                        
        -input_features:    Number of (direct) input_features for the first CNN-layer.
        
        -no_blocks:         Total number of encoder and decoder blocks.
        
        -cat_features:      Boolean; Should the features at each encoder-block be saved after
                            running through the CNN-layer and before pooling to 
                            stack them on top of the input features of each corresponding
                            decoder block? Usually set to true. This introduces skip connections
                            that do not run "the full U" but rather transfer the encoder output
                            directly to the corresponding decoder of equal tensor length.
        
        -use_skip:          Boolean; If set to True, a skip-connection will be used that
                            transfers the input directly to the output layer and stacks it
                            on top of the input tensor of the output layer. 
        
        -trans_layers:      number of CNN-layers to run through before transferring the
                            highest level encoder output to the decoder. This additional
                            group of "transition layers" allow the model to calculate 
                            complex features that take high level input features into account
                            that are very far away from the actual position we want to predict.
                            
        -trans_channels:    Number of output channels of the transition CNNs. Handed to the
                            model as a list, where each element determines the number of output
                            channels of each consecutive CNN layer.
                            
        -trans_kernel_size: Kernel sizes of the transition CNNs. Handed to the
                            model as a list, where each element determines the kernel size
                            of each consecutive CNN layer.
        
        -trans_activation:  Single variable containing a pytorch function that is used
                            within the transititon CNN layers as activation function.
                            
        -output_channels:   Number of channels that should be output by the final layer.
        
        -output_func:       Function to be applied to the final output. In binary classification
                            tasks this is usually a sigmoid function. We could apply a softmax
                            function here, but as the loss function we use (CrossEntropyLoss)
                            already has a softmax function built-in, we will simply be using
                            the identity function.
        
        -use_separate_input: Boolean; Should a separate set of input variables be used?
                             If True the model expects two inputs and the second will be
                             run through a separate input layer that applies a simple
                             linear mapping to reduce the number of channels of the 
                             separate input. 
        
        -si_input_channels:  If use_separate_input is True: Number of channels of the 
                             separate input (i.e. number of additional features)
                             
        -si_output_channels: If use_separate_input is True: Number of channles the 
                             seperate input is reduced to. 
                            
                
    NOTE:   Due to the connections of the model's layers in- and
            outputs, the following must hold for the different channels: 
                
                enc_channels[-1]  = trans_channels[0]
                dec_conv_input_channels[0] = trans_channels[-1] + enc_channels[-1]
                
                if cat_features:
                    dec_conv_input_channels[1:] = enc_channels[:-1] + dec_up_output_channels[:-1]
                else:
                    dec_conv_input_channels[1:] = dec_up_output_channels[:-1]
                                
    '''
    
    def __init__(self, params):
        
        super().__init__()
        
        if not set(['input_features','no_blocks', 'cat_features', 'use_skip',
                    'enc_channels', 'enc_activation','enc_kernel_size', 'enc_no_layers', 
                    'enc_pooling', 'enc_pooling_stride','enc_use_norm',
                    'trans_layers','trans_channels','trans_kernel_size',
                    'trans_activation',
                    'dec_channels', 'dec_up_output_channels', 'dec_conv_input_channels',
                    'dec_activation_func', 'dec_kernel_size', 'dec_no_layers', 'dec_upconv_type', 
                    'dec_upconv_stride', 'dec_use_norm','output_channels', 'output_func',
                    'use_separate_input', 'si_input_channels', 'si_output_channels']).issubset(
                                set(params.keys())):
                            
            print("The params-dict for unet  must contain the following keys: 'input_features','no_blocks', 'cat_features', 'use_skip', 'enc_channels', 'enc_activation','enc_kernel_size', 'enc_no_layers','enc_pooling', 'enc_pooling_stride','enc_use_norm', 'trans_layers','trans_channels','trans_kernel_size', 'trans_activation', 'dec_channels', 'dec_up_output_channels', 'dec_conv_input_channels', 'dec_activation_func', 'dec_kernel_size', 'dec_no_layers', 'dec_upconv_type', 'dec_upconv_stride', 'dec_use_norm','output_channels', 'output_func', 'use_separate_input', 'si_input_channels', 'si_output_channels'.")
            print("Model will be initialized with None.")
            
            return None
        
        else:
            
            super().__init__()
        
            self.use_separat_input = params['use_separate_input']
            
            if self.use_separat_input:
                self.si_cnn = torch.nn.Conv1d(params['si_input_channels'], params['si_output_channels'], kernel_size = 1)
            
            si_features = params['si_output_channels'] if self.use_separat_input else 0
        
            self.no_blocks = params['no_blocks']
            self.cat_features = params['cat_features']
            self.use_skip = params['use_skip']
            
            self.enc_channels = [params['input_features'] + si_features] + params['enc_channels']
            self.enc_block_list = torch.nn.ModuleList([encoder_block(self.enc_channels[i], self.enc_channels[i+1], params['enc_activation'], params['enc_kernel_size'][i], params['enc_no_layers'], params['enc_pooling'], params['enc_pooling_stride'][i], params['enc_use_norm']) for i in range(self.no_blocks)])
            
            self.no_trans_layers = params['trans_layers']
            self.trans_channels = [self.enc_channels[-1]] + params['trans_channels']
            self.trans_cnn_list = torch.nn.ModuleList([torch.nn.Conv1d(self.trans_channels[i], self.trans_channels[i+1], params['trans_kernel_size'][i], padding = int(params['trans_kernel_size'][i]/2), padding_mode = 'zeros') for i in range(params['trans_layers'])])
            self.trans_bn_list = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.trans_channels[i]) for i in range(params['trans_layers'])])
            self.trans_activation = params['trans_activation']
            
            self.dec_channels = [self.enc_channels[-1]] + params['dec_channels']
            self.dec_block_list = torch.nn.ModuleList([decoder_block(self.dec_channels[i],params['dec_up_output_channels'][i], params['dec_conv_input_channels'][i], self.dec_channels[i+1], params['dec_activation_func'], params['dec_kernel_size'][i], params['dec_no_layers'], params['dec_upconv_type'], params['dec_upconv_stride'][i], params['dec_use_norm']) for i in range(self.no_blocks)])
            
            if self.use_skip:
            
                self.out_layer = torch.nn.Conv1d(self.dec_channels[-1] + self.enc_channels[0], params['output_channels'], kernel_size = 3, padding = 1, padding_mode = 'zeros')
            else:
                self.out_layer = torch.nn.Conv1d(self.dec_channels[-1] , params['output_channels'], kernel_size = 3, padding = 1, padding_mode = 'zeros')
            
            self.out_func = params['output_func']
            
    
    def forward(self, x, si = None):
        
        
        if self.use_separat_input:
            si = self.si_cnn(si)
            x = torch.cat([x,si], dim = 1)
        
        if self.use_skip:
            x_copy = x.clone()
        
        enc_outputs = {} # save the encoder outputs seperately to concatenate them later if cat_features = True
        for i in range(self.no_blocks):
            enc_outputs['block' + str(i)], x = self.enc_block_list[i](x)
        
        for i in range(self.no_trans_layers):
            x = self.trans_bn_list[i](x)
            x = self.trans_cnn_list[i](x)
            x = self.trans_activation(x)
   
        for i in range(self.no_blocks):
            x = self.dec_block_list[i](x,self.cat_features, enc_outputs['block' + str(self.no_blocks - (i+ 1))])
        
        if self.use_skip:
            x = torch.cat([x, x_copy], dim = 1)
        #print(x.shape)
        return(self.out_func(self.out_layer(x)))
