import torch
from skimage import io
from torchvision.transforms import transforms

from utils.add_channel import AddChannel
from models.agethnet import AgethNet

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device {device}')

    train_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.57089275, 0.4255322, 0.35874116], [0.24959293, 0.21301098, 0.20608185]),
    ])

    net = AgethNet('models/utk_model_resnet_50.pt', device)
    x = io.imread('imdb_wiki/00/nm0684500_rm387289856_1955-3-4_2007.jpg')
    augmented = train_transform(x).unsqueeze(0).to(device=device)
    predicted_age = net.forward(augmented)
    print(f'predicted_age: {predicted_age}')
    loss_func = torch.nn.MSELoss().to(device=device)
    loss = loss_func(predicted_age, torch.cuda.FloatTensor([52]))
    print(f'loss: {loss}')
    loss.backward()

#     for ethnicity in ['caucasian']:
#         params = net._modules[ethnicity].fc.named_parameters()
#         for name, param in params:
#             if name == 'bias':
#                 print(f'{name}: {param.grad}')

    for param in net.ethnicity_model.parameters():
        print(param.grad)
