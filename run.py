import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from torch import dot, argsort
from torch import sign, count_nonzero, ones, reshape, eye, dot, argsort
from torch.linalg import eig, eigh
from scipy.stats import kendalltau
from torchmetrics.regression import KendallRankCorrCoef
import argparse
from models import MLP
from OrdinalEntropy import ordinal_entropy
from dataset.SAR_dataset import patch_dataset
from torchvision import transforms

lr = 1e-2
epoch = 500
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
mask = list(range(20))
mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
expened_data_num = 400
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'---------DEVICE: {device} -------------')
class ANNRegression(nn.Module):
    def __init__(self, dim=8):
        super(ANNRegression, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.drop_out = nn.Dropout(p=0.)
        # 第一层卷积：输入通道数为20，输出通道数为32，卷积核大小为3x3
        self.conv1 = nn.Linear(20, dim)
        self.conv2 = nn.Linear(dim, dim)
        self.conv3 = nn.Linear(dim, 1)

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)

    def forward(self, x):

        # x = torch.relu(self.conv1(x))
        # x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))
        x = self.leaky_relu(self.conv1(x))
        x = self.drop_out(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.drop_out(x)
        x = self.leaky_relu(self.conv3(x))

        return x


kendalrankloss = KendallRankCorrCoef()


def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None


def centering_matrix(n):
    # centering matrix, projection to the subspace orthogonal
    # to all-ones vector
    return np.eye(n) - np.ones((n, n)) / n


def get_the_subspace_basis(n, verbose=True):
    # returns the orthonormal basis of the subspace orthogonal
    # to all-ones vector
    H = centering_matrix(n)
    s, Zp = np.linalg.eigh(H)
    ind = np.argsort(-s)  # order eigenvalues descending
    s = s[ind]
    Zp = Zp[:, ind]  # second axis !!
    # if (verbose):
    #     print("...forming the Z-basis")
    #     print("check eigenvalues: ", allclose(
    #         s, concatenate((ones(n - 1), [0]), 0)))

    Z = Zp[:, :(n - 1)]
    # if (verbose):
    #     print("check ZZ'=H: ", allclose(dot(Z, Z.T), H))
    #     print("check Z'Z=I: ", allclose(dot(Z.T, Z), eye(n - 1)))
    return Z


def compute_upsets(r, C, verbose=True, which_method=""):
    n = r.shape[0]
    totmatches = count_nonzero(C) / 2
    if (len(r.shape) == 1):
        r = reshape(r, (n, 1))
    e = ones((n, 1)).to(device)
    # Chat = r.dot(e.T) - e.dot(r.T)
    Chat = torch.matmul(r, e.T) - torch.matmul(e, r.T)
    upsetsplus = count_nonzero(sign(Chat[C != 0]) != sign(C[C != 0]))
    upsetsminus = count_nonzero(sign(-Chat[C != 0]) != sign(C[C != 0]))
    winsign = 2 * (upsetsplus < upsetsminus) - 1
    # if (verbose):
    #     print(which_method + " upsets(+): %.4f" %
    #           (upsetsplus / float(2 * totmatches)))
    #     print(which_method + " upsets(-): %.4f" %
    #           (upsetsminus / float(2 * totmatches)))
    return upsetsplus / float(2 * totmatches), upsetsminus / float(2 * totmatches), winsign


def GraphLaplacian(G):
    """
    Input a simlarity graph G and return graph GraphLaplacian
    """
    D = torch.diag(G.sum(dim=1))
    L = D - G

    return L


def get_ulbps_ulbonly(simMat):
    #### input is (lb, unlb) X (lb, unlb) sim matrix
    #### output is (lb + ulb_pslb_tp), ### keep simple for now, just take the closes one
    S = simMat

    n = S.shape[0]
    Z = torch.tensor(get_the_subspace_basis(n, verbose=False)).float().to(device)

    # print(S.shape)
    Ls = GraphLaplacian(S)
    ztLsz = torch.matmul(torch.matmul(Z.T, Ls), Z)
    w, v = eig(ztLsz)
    w = torch.view_as_real(w)[:, 0]
    v = torch.view_as_real(v)[..., 0]

    if torch.is_complex(w):
        print("complex")
        return None

    ind = torch.argsort(w)
    v = v[:, ind]
    r = reshape(torch.matmul(Z, v[:, 0]), (n, 1))

    _, _, rsign = compute_upsets(r, S, verbose=False)

    r_final = rsign * r
    ### r_final is shape [n, 1]
    r_rank = torch.argsort(torch.argsort(r_final.reshape(-1)))

    return r_rank


def ulb_rank(input_feat, lambda_val=-1):
    samples = random.sample(range(0, len(input_feat) - 1), 10)  # random sample 100 features
    input_feat = input_feat[samples]

    p = torch.nn.functional.normalize(input_feat, dim=1)
    # print(p.shape)
    feat_cosim = torch.matmul(p, p.T)
    # print(feat_cosim.shape)
    # exit()

    labels_ulpbs = get_ulbps_ulbonly(feat_cosim)
    labels_ulbpsdornk = labels_ulpbs
    # print(labels_ulbpsdornk.shape)

    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau

    loss_ulb = torch.tensor(0).float().to(device)

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
    batch_unique_targets = torch.unique(ps_ulb_ranked)
    if len(batch_unique_targets) < len(ps_ulb_ranked):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((ps_ulb_ranked == target).nonzero()[:, 0]).item())
        feat_cosim_samp = feat_cosim[:, sampled_indices]
        feat_cosim_samp = feat_cosim_samp[sampled_indices, :]
        ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
    else:
        feat_cosim_samp = feat_cosim
        ps_ulb_ranked_samp = ps_ulb_ranked

    for i in range(len(ps_ulb_ranked_samp)):
        # print("sampling i", i)
        label_ranks = rank_normalised(
            -torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0, 1))
        # print(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
        # exit()
        feature_ranks_ulb0ulb0 = TrueRanker.apply(feat_cosim_samp[i].unsqueeze(dim=0), lambda_val)
        # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

    return loss_ulb, ps_ulb_ranked, samples


def ulb_rank_prdlb(input_feat, lambda_val=-1, pred_inp=None, samples=None):
    # samples = random.sample(range(0, len(input_feat)-1), 10)  # random sample 100 features
    input_feat = input_feat[samples]

    # print(input_feat)
    # pred_inp = pred_inp[samples].detach()
    # print(pred_inp)
    # exit()
    p = input_feat
    # print(p.shape)

    # print(p)
    feat_cosim = -torch.abs(p - p.T)  # !!!!!!!! IMPORTANT!!!!
    ### ORDERING NEEDS TO BE CONSISTENT
    ### LARGER VALUE MUST SIGNAL MORE SIMILAR

    # print(feat_cosim.shape)
    # print(feat_cosim)
    # exit()

    labels_ulpbs = pred_inp.squeeze(-1)
    labels_ulbpsdornk = labels_ulpbs
    # print(labels_ulbpsdornk.shape)
    # exit()
    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau

    loss_ulb = torch.tensor(0).float().to(device)

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
    batch_unique_targets = torch.unique(ps_ulb_ranked)
    if len(batch_unique_targets) < len(ps_ulb_ranked):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((ps_ulb_ranked == target).nonzero()[:, 0]).item())
        feat_cosim_samp = feat_cosim[:, sampled_indices]
        feat_cosim_samp = feat_cosim_samp[sampled_indices, :]
        ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
    else:
        feat_cosim_samp = feat_cosim
        ps_ulb_ranked_samp = ps_ulb_ranked

    for i in range(len(ps_ulb_ranked_samp)):
        # print("sampling i", i)
        label_ranks = rank_normalised(
            -torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0, 1))

        feature_ranks_ulb0ulb0 = TrueRanker.apply(feat_cosim_samp[i].unsqueeze(dim=0), lambda_val)
        # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
        # print(feature_ranks_ulb0ulb0)
        # print(label_ranks)
        # exit()
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

    return loss_ulb

# 设定十折交叉验证
kf = KFold(n_splits=35, shuffle=True, random_state=0)


# 定义训练函数
def train_model(train_loader, val_loader, unlabel_loader, epochs=epoch, lr=lr, dim=8, lambda_val=2, warmup=-1, ulb_w=0.0001, ulb_w2=0.0001):
    train_batchsize = train_loader.batch_size
    val_batchsize = val_loader.batch_size

    #model = ANNRegression(dim=dim)
    model = MLP().to(device)
    #model.initialize()
    model.init_weights()
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 添加学习率衰减（每10个epoch降低学习率）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    # 训练过程
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_loss_oe = 0.0
        running_loss_ulb_0 = 0.0
        running_loss_ulb_1 = 0.0
        running_loss_mse = 0.0
        # 训练循环
        count = 0
        for ((inputs, targets), (inputs_ulb, _)) in zip(train_loader, unlabel_loader):
            if count < 3 and epoch ==0:
                print(inputs_ulb)
                count = count +1
            optimizer.zero_grad()
            inputs = inputs.to(device)
            inputs_ulb = inputs_ulb.to(device)
            targets = targets.to(device)
            # 前向传播
            outputs, features = model(inputs)
            pred_ulb, feature_ulb = model(inputs_ulb)
            # print(outputs.detach().numpy())
            loss_mse = criterion(outputs, targets.reshape(4, 1))
            loss_oe = ordinal_entropy(features, targets.reshape(-1, 1))
            if ulb_w > 0:
                # assert False, "Not Done yet"
                loss_ulb_0_unweighted, ft_rank, samples = ulb_rank(feature_ulb, lambda_val)
                loss_ulb_0 = loss_ulb_0_unweighted * ulb_w
                # print(ft_rank)
                if epoch > warmup:
                    loss_ulb_1 = ulb_rank_prdlb(pred_ulb, lambda_val, pred_inp=ft_rank, samples = samples) * ulb_w2
                else:
                    loss_ulb_1 = loss_oe * 0
            else:
                loss_ulb_0 = loss_oe * 0
                loss_ulb_1 = loss_oe * 0

            loss = loss_mse + loss_oe * 0.01 + loss_ulb_0 + loss_ulb_1


            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_oe += loss_oe.item()
            running_loss_ulb_0 += loss_ulb_0.item()
            running_loss_ulb_1 += loss_ulb_1.item()
            running_loss_mse += loss_mse.item()

        # 更新学习率
        # scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Loss_oe: {running_loss_oe / len(train_loader)}, Loss_ulb0: {running_loss_ulb_0 / len(train_loader)}, Loss_ulb1: {running_loss_ulb_1 / len(train_loader)}, Loss_mse: {running_loss_mse / len(train_loader)} ,LR: {scheduler.get_last_lr()[0]}")

    # 验证过程
    model.eval()
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, features = model(inputs)

            val_predictions.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_predictions = np.concatenate(val_predictions, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    return model, val_predictions, val_targets


# 进行十折交叉验证
mse_scores = []
rmse_scores = []
r2_scores = []

total_val_predictions = []
total_val_targets = []

true_data_label = np.load('../original_features_labels_400.npz')
print(true_data_label['features'].shape)
data = true_data_label['features']
lbl = true_data_label['labels']
expended = np.load('../features_labels_400.npz')
print('!!!', expended['features'].shape)
data = np.concatenate((data, expended['features'][0: 0 + expened_data_num, :]))
lbl = np.concatenate((lbl, expended['labels'][0: 0 + expened_data_num]))
true_lbl = lbl[0: 35]
expended_lbl = lbl[35:]
true_data = data[0: 35, :]
expended_data = data[35:, :]
unlabeled_data = np.load('../unlabeled_features.npz')['features']


for fold, (train_index, test_index) in enumerate(kf.split(true_data)):
    print(f'---------Fold: {fold}  test_index: {test_index}------------')
    features_id = mask
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = lbl[train_index], lbl[test_index]
    index1 = np.asarray(list(range(0, expened_data_num, 2)))
    index2 = np.asarray(list(range(1, expened_data_num, 2)))
    X_train1 = np.concatenate((X_train, expended_data[index1]))[:, features_id]
    y_train1 = np.concatenate((y_train, expended_lbl[index1]))
    X_train2 = np.concatenate((X_train, expended_data[index2]))[:, features_id]
    y_train2 = np.concatenate((y_train, expended_lbl[index2]))
    x_test, y_test = data[test_index][:, features_id], lbl[test_index]
    dataset1 = TensorDataset(torch.tensor(X_train1, dtype=torch.float32), torch.tensor(y_train1, dtype=torch.float32))
    dataset2 = TensorDataset(torch.tensor(X_train2, dtype=torch.float32), torch.tensor(y_train2, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader1 = DataLoader(dataset1, drop_last=True, batch_size=4)
    train_loader2 = DataLoader(dataset2, drop_last=True, batch_size=4)
    val_loader = DataLoader(val_dataset, drop_last=True, batch_size=1)
    empty_labels = torch.zeros(unlabeled_data.shape[0])
    unlabel_dataset = TensorDataset(torch.tensor(unlabeled_data, dtype=torch.float32), empty_labels)
    unlabel_loader = DataLoader(unlabel_dataset, drop_last=True, batch_size=100, shuffle=True)


    # 训练模型并返回验证集上的预测结果和目标值
    model, val_predictions1, val_targets = train_model(train_loader1, val_loader, unlabel_loader, dim=8, warmup=-1)
    model, val_predictions2, val_targets = train_model(train_loader2, val_loader, unlabel_loader, dim=8, warmup=-1)


    total_val_targets += val_targets.tolist()
    total_val_predictions += ((val_predictions1 + val_predictions2) / 2).tolist()
    #total_val_predictions += val_predictions1.tolist()

# 计算验证集上的均方误差 (MSE)
mse = mean_squared_error(total_val_targets, total_val_predictions)
rmse = np.sqrt(mse)
rmse_scores.append(rmse)
mse_scores.append(mse)

# 计算 R2 指标
r2 = r2_score(total_val_targets, total_val_predictions)
r2_scores.append(r2)
fig, ax = plt.subplots()
plt.title("CLSS", loc="center")
total_val_targets = np.asarray(total_val_targets)
total_val_predictions = np.asarray(total_val_predictions)
ax.plot(range(total_val_targets.shape[0]), total_val_targets, label='gt')
ax.plot(range(total_val_targets.shape[0]), total_val_predictions, label='pred')
ax.set(xlabel='Sample', ylabel=f'lai')
ax.legend()
plt.show()
print(f"十折交叉验证的平均MSE: {mse}")
print(f"十折交叉验证的平均R²: {r2}")
with open(f'./result/result_{epoch}.txt', 'w') as f:
    f.write(f'MSE:{mse}, R2:{r2}')
    f.close()
