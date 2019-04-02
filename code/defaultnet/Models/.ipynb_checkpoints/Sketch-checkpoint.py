from .Base import BaseModel
from ..Network import Head,Backbone
from ..Losses.yolo_loss import YoloLoss as loss

class SketchModel(BaseModel):
    def __init__(self, num_classes=20, weights_file=None, input_channels=3, train_flag=True, test_args=None):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.nloss = 1
        self.train_flag = train_flag

        self.loss = None
        self.postprocess = None

        num_anchors_list = [2,5,8]
        in_channels_list = [512, 256, 128]

        self.backbone = Backbone.Darknet53()
        self.head = Head.Yolov3(num_classes, in_channels_list, num_anchors_list)

        if weights_file is not None:
            self.load_weights(weights_file, clear)
        else:
            self.init_weights(slope=0.1)

    def _forward(self, x):
        middle_feats = self.backbone(x)
        features = self.head(middle_feats)
        loss_fn = loss
        
        """
        generate loss and postprocess
        """
        if self.train_flag == 1: # train
            if self.loss is None:
                self.loss = [] # for training

                for idx in range(self.nloss):
                    reduction = float(x.shape[2] / features[idx].shape[2]) # n, c, h, w
                    self.loss.append(loss_fn(self.num_classes, reduction, self.seen, head_idx=idx))
        elif self.train_flag == 2: # test
            if self.postprocess is None:
                self.postprocess = [] # for testing

                conf_thresh = self.test_args['conf_thresh']
                network_size = self.test_args['network_size']
                labels = self.test_args['labels']
                for idx in range(self.nloss):
                    reduction = float(x.shape[2] / features[idx].shape[2]) # n, c, h, w
#                     cur_anchors = [self.anchors[ii] for ii in self.anchors_mask[idx]]
#                     cur_anchors = [(ii[0] / reduction, ii[1] / reduction) for ii in cur_anchors] # abs to relative
#                     self.postprocess.append(vnd.transform.Compose([
#                         vnd.transform.GetBoundingBoxes(self.num_classes, cur_anchors, conf_thresh),
#                         vnd.transform.TensorToBrambox(network_size, labels)
# ]))

        return features

#     def modules_recurse(self, mod=None):
#         """ This function will recursively loop over all module children.
#         Args:
#             mod (torch.nn.Module, optional): Module to loop over; Default **self**
#         """
#         if mod is None:
#             mod = self

#         for module in mod.children():
#             if isinstance(module, (nn.ModuleList, nn.Sequential, backbone.Darknet53,
#                 backbone.Darknet53.custom_layers, head.Yolov3, head.Yolov3.custom_layers)):
#                 yield from self.modules_recurse(module)
#             else:
#                 yield module