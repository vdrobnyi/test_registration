import torch
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

# Copypasted from https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch

"""
model by dhiraj inspried from Charles
"""
class PCAutoEncoder(nn.Module):
    """ Autoencoder for Point Cloud
    Input:
    Output:
    """

    def __init__(self, point_dim, num_points, hidden_size, decode=True):
        super(PCAutoEncoder, self).__init__()

        self.decode = decode
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(in_channels=point_dim + 3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=self.hidden_size, kernel_size=1)

        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_points * 3)

        # batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn_np = nn.BatchNorm1d(num_points)
        self.bn_hs = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        point_dim = 3
        x = x.reshape(batch_size, num_points, 6).transpose(1, 2)

        # encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # do max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.hidden_size)
        # get the global embedding
        global_feature = x

        if self.decode:
            # decoder
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            reconstructed_points = self.fc3(x)

            # do reshaping
            reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)
        else:
            reconstructed_points = None

        return reconstructed_points, global_feature


# Visualise input/output of auto-encoder
def draw_comparison(orig, pred):
    orig = np.array(orig[0].cpu().detach().squeeze())
    pred = np.array(pred[0].cpu().detach().squeeze())
    o3d_orig = o3d.geometry.PointCloud()
    o3d_pred = o3d.geometry.PointCloud()

    o3d_orig.points = o3d.utility.Vector3dVector(orig)
    o3d_orig.paint_uniform_color([0.2, 0.8, 0.2])

    o3d_pred.points = o3d.utility.Vector3dVector(pred)
    o3d_pred.paint_uniform_color([0.2, 0.2, 0.8])

    o3d.visualization.draw_geometries([o3d_orig, o3d_pred])


class PointEncoder:
    def __init__(self, net_path, out_features=16):
        self.net = PCAutoEncoder(3, 1000, out_features)
        self.load_net(net_path)

    def load_net(self, path):
        self.net.load_state_dict(torch.load(path))

    def encode_points(self, query_points, pcd, radius, num_points, split=False):
        tree = KDTree(pcd)

        sub_pcds = []
        chunk_size = 10000
        query_points = [query_points[i:i+chunk_size] for i in range(0, len(query_points), chunk_size)]
        for chunk in query_points:
            sub_pcds_tmp = PointEncoder.__crop_inputs_tree(chunk, tree, radius)
            sub_pcds_tmp = [p - np.mean(p, axis=0) for p in sub_pcds_tmp]
            sub_pcds_tmp = [p[np.random.choice(p.shape[0], size=num_points, replace=True)] for p in sub_pcds_tmp]
            sub_pcds.extend(sub_pcds_tmp)

        dataset = PointCloudDataset(sub_pcds)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

        self.net.eval()
        device = torch.device("cuda")
        self.net.to(device)
        features = None
        for i, data in enumerate(data_loader):
            points = data.clone().detach()
            points = points.transpose(2, 1)
            points = points.to(device)

            _, latent_vector = self.net(points)
            latent_vector = np.array(latent_vector.cpu().detach().float(), dtype=np.float32)
            if features is None:
                features = latent_vector
            else:
                features = np.concatenate([features, latent_vector])
        return np.array(features)


class PointCloudDataset(Dataset):
    """ Point Cloud Dataset """
    def __init__(self, data_list):
        self.dataset = data_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        # convert to pytorch tensor
        point_set = torch.from_numpy(data).float()  # convert to float32

        return point_set


from chamfer_distance import ChamferDistance
        
def train_encoder(train_dataset, test_dataset=None):
    device = torch.device("cuda")
    net = PCAutoEncoder(3, 1000, 64).to(device)
    net = net.train()

    criterion = ChamferDistance
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    sheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.9)

    batch_size = 1024
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data_loader = None
    if test_dataset is not None:
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(5000):
        net.train()
        for i, data in enumerate(data_loader):
            points = data
            points = points.transpose(2, 1)
            points = points.to(device)

            optimizer.zero_grad()

            reconstructed_points, latent_vector = net(points)

            bs = points.shape[0]
            points = points.reshape(bs, 1000, 6)
            reconstructed_points = reconstructed_points.transpose(1, 2)
            points = points[:, :, :3]
            loss = (criterion(points, reconstructed_points)[0].mean() + criterion(reconstructed_points, points)[0].mean()) * 0.5  # calculate loss

            if i % 10000000 == 0:
                print(f"Epoch: {epoch}, Iteration#: {i}, Train Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

        if epoch > 0 and epoch < 100 and epoch % 10 == 0:
            sheduler.step()
        
        if epoch % 10 != 0:
            continue

        with torch.no_grad():
            if test_data_loader is not None:
                net.eval()
                total_loss = 0
                for i, data in enumerate(test_data_loader):
                    points = data
                    points = points.transpose(2, 1)
                    points = points.to(device)

                    reconstructed_points, latent_vector = net(points)  # perform training

                    bs = points.shape[0]
                    points = points.reshape(bs, 1000, 6)
                    reconstructed_points = reconstructed_points.transpose(1, 2)
                    points = points[:, :, :3]
                    loss = (criterion(points, reconstructed_points)[0].mean() + criterion(reconstructed_points, points)[0].mean()) * 0.5  # calculate loss#criterion(points, reconstructed_points) # + criterion(reconstructed_points, points)  # calculate loss
                    total_loss += loss.item()
                print(f"Epoch: {epoch}, Test Loss: {total_loss / len(test_data_loader)}")

    return net


def train():
    data_path = "train_ae_10m.npy"

    data_list = np.load(data_path, allow_pickle=True)
    dataset = PointCloudDataset(data_list)

    net = train_encoder(dataset, dataset)
    torch.save(net.state_dict(), "net.ptr")


if __name__ == "__main__":
    train()