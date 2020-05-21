class CascadeFeauture:
    def __init__(self, mode, config, model_dir=None):
        self.config = config
        ###self.config = Config()
        self.model_mask = MaskRCNN(mode, config, model_dir)
        self.model_mask.load_weights(model_dir, by_name=True)
        self.model_mask_keras = self.model_mask.keras_model
        self.output = Model(inputs=self.model_mask_keras.inputs, output= \
            [self.model_mask_keras.get_layer('fpn_p5').output, \
             self.model_mask_keras.get_layer('roi_align_classifier').output, \
             self.model_mask_keras.get_layer('mrcnn_class_logits').output])
