from model.resnet import ResNet18_Bottom, ResNet18_Top
from model.vgg import VGG16_Bottom, VGG16_Top
from model.MixText import MixText_Bottom, MixText_Top



def get_model(args):
    if args.model_type == 'resnet18':
        bottom_model, top_model = ResNet18_Bottom(args), ResNet18_Top(args)
    elif args.model_type == 'vgg16':
        bottom_model, top_model = VGG16_Bottom(args), VGG16_Top(args)
    elif args.model_type == 'mixtext':
        bottom_model, top_model = MixText_Bottom(args), MixText_Top(args)
    else:
        raise ValueError(f'No model named {args.model_type}!')
    return bottom_model, top_model