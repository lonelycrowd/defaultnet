### Structure:
- Models -- envelopes different parts (backbone, head, loss) to resulting model.
- Network -- this folder contains three subfolders:

    1) Backbone  -- each file here contains a subclass of nn.Module, which can be used as a backbone, usually it is some classifier model with cuted last layers.
    
    2) Head -- each file here contains a subclass of nn.Module, which can be used as a head, i.e. usually this part is one of determining for the model name.
    
    3) Blocks -- these files contains building blocks for the models, that can be usefull in construction of any model.
        3.1) Stages -- greater blocks.
    
- Losses -- it is clear what is in the folder?. Files should be named as "model name"_loss.py
- Machine -- there is two main files here:
    
    1) train_machine.py -- this script contains everything to train model from "Model",  end-to-end
    
    2) test_machine.py --  this script is for evaluating trained model on some dataset, in future it should be able to save results in many standard formats.
    

- Data -- tools to work load different datasets to the standard format which will be able to be forwarded in our models. Has to subfolders:

    1) Datasets -- subfolders should correspond to different standard datasets (VOC_dataset, wider_dataset,...)
    
    2) Augmentation -- different transforms, helping to imitate greater dataset
    
    3) Basics -- basic utils

