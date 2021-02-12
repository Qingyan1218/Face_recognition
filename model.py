import torch
import torch.nn as nn
import torchvision.models as models
from cbam_module import resnet18_cbam, resnet34_cbam, resnet50_cbam, resnet101_cbam, resnet152_cbam

class ResnetTriplet(nn.Module):
    def __init__(self, model_version = '18', embedding_dimension=128, pretrained=False):
        super(ResnetTriplet, self).__init__()
        if model_version == '18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_version == '34':
            self.model = models.resnet34(pretrained=pretrained)
        elif model_version == '50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model_version == '101':
            self.model = models.resnet101(pretrained=pretrained)
        elif model_version == '152':
            self.model = models.resnet152(pretrained=pretrained)
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        alpha = 10
        embedding = embedding * alpha

        return embedding

class ResnetCBAMTriplet(nn.Module):
    def __init__(self, model_version = '18', embedding_dimension=128, pretrained=False):
        super(ResnetCBAMTriplet, self).__init__()
        if model_version == '18':
            self.model = resnet18_cbam(pretrained=pretrained)
        elif model_version == '34':
            self.model = resnet34_cbam(pretrained=pretrained)
        elif model_version == '50':
            self.model = resnet50_cbam(pretrained=pretrained)
        elif model_version == '101':
            self.model = resnet101_cbam(pretrained=pretrained)
        elif model_version == '152':
            self.model = resnet152_cbam(pretrained=pretrained)

        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        alpha = 10
        embedding = embedding * alpha

        return embedding

if __name__=='__main__':
    x = torch.rand(4,3,256,256)
    print(x.shape)
    model = ResnetCBAMTriplet('18', 128)
    result = model(x)
    print(result.shape)

