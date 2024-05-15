
# --- Imports --- #
import torch
import torch.nn.functional as F
from torchvision.models import vgg16

# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)

def Perceptual_loss():
    # --- Define the perceptual loss network --- #
    vgg_model = vgg16(pretrained=True).features[:16]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg_model = vgg_model.to(device)
    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    for param in vgg_model.parameters():
        param.requires_grad = False

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    perceptual_loss = loss_network
    return perceptual_loss