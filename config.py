from torchvision import transforms

batch_size = 16
num_workers = 4
input_resize = 224
fine_tune_num_epochs = 25
num_epochs = 50
num_new_hidden_neurons = 1024
fix_pretrained_weights = False
SGD = False
learning_rate = 0.001

# change this to reflect data
num_metadata = 5
num_measures = 5

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_resize),
        transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


