# imports
import os
import time
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

Work_space_path = "/homes/nfs/caslab_bs/Desktop/BingYu/Conv_Classification/"

class Img_Dataset(Dataset):
    def __init__(self,mode):
        super().__init__()
        if type(mode) == type(self):
            self.mode = mode.mode
            self.transform = mode.transform
            self.datas = mode.datas
            self.data_root_path = mode.data_root_path
        else:
            self.mode = mode
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.datas = []

            self.data_root_path = Work_space_path+"different_size_data/"+mode+"_set/"
            files = os.listdir(self.data_root_path)
            i = 0
            for file in files:
                if (i+1)%1000 == 0:
                    print(f"[{i+1}/{len(files)}] images are readed.",end='\r')
                i +=1             
                img = Image.open(self.data_root_path+file)
                img = img.resize((64,64))
                img_tensor = self.transform(img)
                img_tensor.requires_grad=True
                label = 0
                if "dog" in file:
                    # dog img
                    label = 0
                else:
                    # cat img 
                    label = 1
                signal_data = (img_tensor,label)
                self.datas.append(signal_data)
    
    def __getitem__(self, index):
        if index < len(self):
            return self.datas[index]
        else:
            return self.datas[0]
        
    def __len__(self):
        return len(self.datas)
    
    def delete(self,need_delete_index):
        del self.datas[need_delete_index]
    
class CNN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.cv_bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.cv_bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.cv_bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.fc_bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # CNN layer
        x = self.cv_bn1(x)
        x = self.pool(F.relu(self.conv2(x)))    # CNN layer
        x = self.cv_bn2(x)
        x = self.pool(F.relu(self.conv3(x)))    # CNN layer
        x = self.cv_bn3(x)
        x = torch.flatten(x, 1)                 # flatten all dimensions except batch
        x = F.relu(self.fc1(x))                 # fully connected layer            
        x = self.fc_bn1(x)
        x = F.relu(self.fc2(x))                 # fully connected layer            
        x = self.fc_bn2(x)
        x = F.relu(self.fc3(x))                 # fully connected layer            
        x = self.fc_bn3(x)
        x = F.relu(self.fc4(x))                 # fully connected layer            
        x = self.fc_bn4(x)
        x = self.fc5(x)                         # fully connected layer    
        return x

    def cal_loass(self, pred_o,true_o):
        return self.criterion(pred_o,true_o)

def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model,train_dataloader,train_set, vaild_set,test_set):
    
    model_path = Work_space_path+"model/model.pth"
    model.to(get_device())
    num_epochs = 20
    learing_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learing_rate,momentum=0.5)
    best_acc = 0
    batch_loss_log = []
    vaild_accuracy_log = []
    test_accuracy_log = []
    train_accuracy_log = []
    batch_loss = 0.6
    max = 0
    for epoch in range(num_epochs):
        # train
        model.train()
        avg_loss = 0
        for i,data in enumerate(train_dataloader):
            inputs,labels = data
            inputs,labels = inputs.to(get_device()), labels.to(get_device())
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = model.cal_loass(outputs,labels)
            avg_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            if (i+1)%100 == 0:
                if i> max:
                    max = i
                avg_loss = avg_loss/100
                print(f"epoch: [{epoch+1:-3d}/{num_epochs}] batch: [{i+1:-4d}/{len(train_set)/train_dataloader.batch_size:.0f}] Loss: {avg_loss:.4f}")
                batch_loss_log.append(avg_loss.item())
                avg_loss = 0

        model.eval()
        train_accuracy = test(model,train_set,"train")
        train_accuracy_log.append(train_accuracy)

        model.eval()
        vaild_accuracy = test(model,vaild_set,"vaild")
        vaild_accuracy_log.append(vaild_accuracy)

        model.eval()
        test_accuracy = test(model,test_set,"test")
        test_accuracy_log.append(test_accuracy)

        if vaild_accuracy > best_acc:
            best_acc = vaild_accuracy
            torch.save(model, model_path)
            print("Saving model...")
        else:
            print(f"One epoch complete with vaild set accuracy {vaild_accuracy:.2f}%")

    plot(batch_loss_log, vaild_accuracy_log,test_accuracy_log,train_accuracy_log,train_dataloader.batch_size)

def test(model,dataset,mode):
    with torch.no_grad():
        accuracy = 0
        cat_acc = 0
        cat_total = 0
        dog_acc = 0
        dog_total = 0
        for i in range(len(dataset)):
            inputs,labels = dataset.__getitem__(i)
            inputs = inputs.unsqueeze(0)

            if labels == 0:
                cat_total +=1
            else:
                dog_total +=1
            outputs = model(inputs)

            _,prediction = torch.max(outputs,1)
            if prediction == labels:
                accuracy +=1
                if labels == 0:
                    cat_acc +=1
                else:
                    dog_acc +=1
        accuracy = float(accuracy/len(dataset)*100)
        print("=============================================")
        print(f'{mode} set: Overall accuracy {accuracy:.2f}%')
        print(f"Dog accuracy {float(dog_acc)/dog_total*100:.2f}%")    
        print(f"Cat accuracy {float(cat_acc)/cat_total*100:.2f}%.")
        print("=============================================")
    return accuracy

def plot(batch_loss_log, vaild_accuracy_log,test_accuracy_log,train_accuracy_log,batch_size):
    fig,ax = plt.subplots(2,1)
    plt.tight_layout()

    plt.subplot(2,1,1)
    plt.title(f"Cross Entropy Loss per 100 batch (batch size: {batch_size})")
    plt.plot(range(len(batch_loss_log)),batch_loss_log,'r')
    plt.grid(True)


    plt.subplot(2,1,2)
    plt.title("Accuracy(%) per epoch")
    plt.plot(range(len(train_accuracy_log)),train_accuracy_log,'b')
    plt.plot(range(len(vaild_accuracy_log)),vaild_accuracy_log,'darkorange')
    plt.plot(range(len(test_accuracy_log)),test_accuracy_log,'r')
    plt.legend(labels = ["train set","vaild set","test set"],loc = "best")
    plt.grid(True)



    with open(Work_space_path+'loss_log.txt', 'w') as f:
        for los in batch_loss_log:
            f.write(f"{los}\n")

    with open(Work_space_path+'vaild_acc_log.txt', 'w') as f:
        for acc in vaild_accuracy_log:
            f.write(f"{acc}\n")
    
    with open(Work_space_path+'test_acc_log.txt', 'w') as f:
        for acc in test_accuracy_log:
            f.write(f"{acc}\n")
    
    with open(Work_space_path+'train_acc_log.txt', 'w') as f:
        for acc in train_accuracy_log:
            f.write(f"{acc}\n")

    return

def main():
    print("Loading training data...")
    batch_size = 20
    train_set = Img_Dataset("train")
    print("Loading training data complete.")

    print("Loading testing data...")
    test_set = Img_Dataset("test")
    print("Loading testing data complete.")

    print("Creating vaildation set...")
    vaild_set_percent = 0.2
    train_set,vaild_set = torch.utils.data.random_split(
        dataset = train_set,
        lengths = [int((1-vaild_set_percent)*len(train_set)),int(vaild_set_percent*len(train_set))],
        generator = torch.Generator().manual_seed(int(time.time()))
    )
    train_dataloader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    print("Creating vaildation set complete.")

    model = CNN_Network()
    print("Start training...")
    train(model,train_dataloader,train_set,vaild_set,test_set)
    print("Training complete.")
    model = CNN_Network()
    model=torch.load(Work_space_path +"./model/model.mod")

    test(model,test_set,"test")


if __name__ == "__main__":
    main()
