import os
import argparse
import gc
import numpy as np
import torch
from torchvision import transforms
from model import Model, Model2Layers, ModelNoClassifier, ModelNoClassifier2Layers
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr as corr


parser = argparse.ArgumentParser()

parser.add_argument("--train_img_dir", type=str, help="Path to the directory containing training images", required = True)
parser.add_argument("--test_img_dir", type=str, help="Path to the directory containing test images", required = True)
parser.add_argument("--test_pred_dir", default = "./", type=str, help="Path to the directory where to save the predictions for test images")

parser.add_argument("--lh_fmri_path", type=str, help="Path to numpy file containing fmri responses to training images for the left hemisphere", required = True)
parser.add_argument("--rh_fmri_path", type=str, help="Path to numpy file containing fmri responses to training images for the right hemisphere", required = True)

parser.add_argument("--num_v_lh", type=int, help="Number of vertices in the left hemisphere", required = True)
parser.add_argument("--num_v_rh", type=int, help="Number of vertices in the right hemisphere", required = True)

parser.add_argument("--device", default="cuda", type=str, help="Device to use: cuda or cpu")

parser.add_argument("--lr", default = 0.0001, type=float, help="Learning rate")
parser.add_argument("--num_epochs", default = 150, type=int, help="Maximum number of epochs")
parser.add_argument("--batch_size", default = 64, type=int, help="Batch size")
parser.add_argument("--perc_train", type=float, default = 0.9, help="Percentage of training images within the training directory (the remaining are validation images)")
parser.add_argument("--ptc_counter", default = 5, type=int, help="Patience counter (Max. number of epochs without improvements)")
parser.add_argument("--lr_drop", default = 0, type=int, help="Learning rate drop rate")
parser.add_argument("--log_freq", default = 30, type=int, help="Logging frequency (in number of batches)")


'''
python train.py --train_img_dir=/media/data1/algonauts_2023_challenge_data/subj01/training_split/training_images 
--test_img_dir=/media/data1/algonauts_2023_challenge_data/subj01/test_split/test_images 
--lh_fmri_path=/media/data1/algonauts_2023_challenge_data/subj01/training_split/training_fmri/lh_training_fmri.npy 
--rh_fmri_path=/media/data1/algonauts_2023_challenge_data/subj01/training_split/training_fmri/rh_training_fmri.npy
--num_v_lh=19004 
--num_v_rh=20544
'''

'''
python train.py --train_img_dir=/media/data1/algonauts_2023_challenge_data/subj01/training_split/training_images --test_img_dir=/media/data1/algonauts_2023_challenge_data/subj01/test_split/test_images --lh_fmri_path=/media/data1/algonauts_2023_challenge_data/subj01/training_split/training_fmri/lh_training_fmri.npy --rh_fmri_path=/media/data1/algonauts_2023_challenge_data/subj01/training_split/training_fmri/rh_training_fmri.npy --num_v_lh=19004 --num_v_rh=20544
'''

'''LEARNING_RATE = 0.0001
NUM_EPOCHS = 150
DEVICE = "cuda:1"
PATIENCE = 5
LOG_FREQ = 30
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64'''


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform, lh_fmri, rh_fmri):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.lh_fmri = lh_fmri
        self.rh_fmri = rh_fmri

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(args.device)
        return img, self.lh_fmri[idx], self.rh_fmri[idx]



def train(args):
    ######################################## DATA PREPARATION ########################################
    
    print("Learning Rate (LR): ", args.lr)
    
    data_dir = '/media/data1/algonauts_2023_challenge_data'

    ### Device setting 
    device = torch.device(args.device if torch.cuda.is_available() else "cpu") # if DEVICE="cpu" it still works

    #args = argObj(data_dir, "./submission", subj)

    ### Fmri preparation
    lh_fmri = np.load(args.lh_fmri_path)
    rh_fmri = np.load(args.rh_fmri_path)

    print("Training stimulus images × LH vertices")
    print("LH training fMRI data shape:")
    print(lh_fmri.shape)
    
    print("(Training stimulus images × RH vertices)")
    print("\nRH training fMRI data shape:")
    print(rh_fmri.shape)


    train_img_dir = args.train_img_dir
    test_img_dir = args.test_img_dir

    # Create lists with all training and test image file names sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
    print("Training images: " + str(len(train_img_list)))
    print("Test images: " + str(len(test_img_list)))

    rand_seed = 5 #@param
    np.random.seed(rand_seed)

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) * args.perc_train))
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    # Assign 90% of the shuffled stimulus images to the training partition,
    # and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))

    print("Training stimulus images: " + format(len(idxs_train)))
    print("\nValidation stimulus images: " + format(len(idxs_val)))
    print("\nTest stimulus images: " + format(len(idxs_test)))


    transform = transforms.Compose([
        transforms.Resize((224,224)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
    ])

    batch_size = args.batch_size 
    print("Batch size: ", batch_size)
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_val = rh_fmri[idxs_val]

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform, lh_fmri_train, rh_fmri_train),
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform, lh_fmri_val, rh_fmri_val),
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform, lh_fmri_train, rh_fmri_train),
        batch_size=batch_size
    )


    del lh_fmri, rh_fmri
    gc.collect()

    # Model 1 Layer
    #model = Model(num_lh = args.num_v_lh, num_rh = args.num_v_rh)

    # Model 2 Layers
    #model = Model2Layers(num_lh = args.num_v_lh, num_rh = args.num_v_rh)
    
    # Model No classifier 1 Layer
    model = ModelNoClassifier(num_lh = args.num_v_lh, num_rh = args.num_v_rh) 

    # Model No classifier 2 Layers, 5000
    #model = ModelNoClassifier2Layers(num_lh = args.num_v_lh, num_rh = args.num_v_rh)
    

    print(model)

    criterion = torch.nn.MSELoss()
    print("MSE is applied")

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    print("Adam on model.parameters()")

    # The following rows can be uncommented to use AdamW and/or LR schedluer

    ### optimizer = torch.optim.AdamW(params = model.parameters(), lr = args.lr)
    ### print(f"AdamW on model.parameters(), no weight decay")
    ### lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print("No AdamW")
    print("No LR scheduler")


    model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
    model.train(True)


    patience_counter = args.ptc_counter
    best_lh_corr = -100
    best_rh_corr = -100

    total_batches = (int)(np.ceil((args.perc_train * len(train_img_list)) / args.batch_size))

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        model.train(True)

        loss_lh_avg = 0
        loss_rh_avg = 0

        # Training loop
        for i, (images, lh_fmri, rh_fmri) in enumerate(train_imgs_dataloader):
            
            images = images.to(device)
            lh_fmri = lh_fmri.to(device)
            rh_fmri = rh_fmri.to(device)

            # Forward pass + backpropagation + loss
            outputs = model(images)

            loss_lh = criterion(outputs[0], lh_fmri) #MSE
            loss_rh = criterion(outputs[1], rh_fmri) #MSE

            # Uncomment the following 2 rows and comment the previous two to use RMSE instead of MSE
            ### loss_lh = torch.sqrt(criterion(outputs[0], lh_fmri)) #RMSE
            ### loss_rh = torch.sqrt(criterion(outputs[1], rh_fmri)) #RMSE

            loss = loss_lh + loss_rh

            optimizer.zero_grad()
            loss.backward()

            # Update model parameters
            optimizer.step()

            loss_lh_avg += loss_lh.item()
            loss_rh_avg += loss_rh.item()

            if(i % args.log_freq == 0) :
                print(f"Batch: {i} / {total_batches}")
                print("Avg loss lh: ", loss_lh_avg / args.log_freq)
                print("Avg loss rh: ", loss_rh_avg / args.log_freq)
                loss_lh_avg = 0
                loss_rh_avg = 0

            
            del loss_lh
            del loss_rh 

            del loss
            del images, lh_fmri, rh_fmri


        ################################# CORRELATION CALCULATION #################################

        model.eval()

        print("Model in eval")

        lh_fmri_val_pred_tmp = []
        rh_fmri_val_pred_tmp = []

        for images, fmri_val_lh_label, fmri_val_rh_label in val_imgs_dataloader:
            out = model(images)

            lh_fmri_val_pred_tmp.append(torch.from_numpy(out[0].detach().cpu().numpy()))
            rh_fmri_val_pred_tmp.append(torch.from_numpy(out[1].detach().cpu().numpy()))
            del out, images, 

        # Empty correlation array of shape: (LH vertices)
        lh_fmri_val_pred = torch.Tensor(len(lh_fmri_val_pred_tmp), len(lh_fmri_val_pred_tmp[0]))
        torch.cat(lh_fmri_val_pred_tmp, out = lh_fmri_val_pred)

        lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh_fmri_val_pred.shape[1])):
            lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_fmri_val_pred = torch.Tensor(len(rh_fmri_val_pred_tmp), len(rh_fmri_val_pred_tmp[0]))
        torch.cat(rh_fmri_val_pred_tmp, out = rh_fmri_val_pred)

        rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh_fmri_val_pred.shape[1])):
            rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]

        lh_mean_corr = np.mean(lh_correlation)
        rh_mean_corr = np.mean(rh_correlation)

        print("Mean lh corr:", lh_mean_corr)
        print("Mean rh corr:", rh_mean_corr)

        ################################################################################


        if(lh_mean_corr < best_lh_corr and 
           rh_mean_corr < best_rh_corr):
            patience_counter += 1
        
        if(lh_mean_corr >= best_lh_corr or 
           rh_mean_corr >= best_rh_corr):
            patience_counter = 0
            best_lh_corr = lh_mean_corr
            best_rh_corr = rh_mean_corr

            print("Saving the best model")
            os.makedirs(f"./checkpoints", exist_ok=True)
            torch.save(model, f"./checkpoints/best.pth")
            
            # Saves the best correlation value 
            with open(f"./checkpoints/log.txt", "a+") as text_file:
                text_file.write(f"Epoch: [{epoch}]: best lh_corr: {best_lh_corr} --- best rh_corr: {best_rh_corr} \n")



            ############################################### TEST PREDICTIONS ###############################################
            lh_fmri_test_pred = []
            rh_fmri_test_pred = []
            
            for images, _, _ in test_imgs_dataloader:
                print(f"Batch size: {images.shape}")
                out = model(images)
                lh_fmri_test_pred.append(torch.from_numpy(out[0].detach().cpu().numpy()))
                rh_fmri_test_pred.append(torch.from_numpy(out[1].detach().cpu().numpy()))
                del out, images, 

            lh_fmri_test_pred = np.vstack(lh_fmri_test_pred)
            rh_fmri_test_pred = np.vstack(rh_fmri_test_pred)

            print("SHAPE OF lh_fmri_test_pred ", lh_fmri_test_pred.shape)
            print("SHAPE OF rh_fmri_test_pred ", rh_fmri_test_pred.shape)

            lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
            rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)

            np.save(os.path.join(args.test_pred_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
            np.save(os.path.join(args.test_pred_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

        

        if patience_counter >= args.ptc_counter:
            patience_counter = 0
            print("Patience counter reached value ", args.ptc_counter)
            break


if __name__ == "__main__":
    args = parser.parse_args()

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    train(args)



