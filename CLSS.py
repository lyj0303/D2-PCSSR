import random
import torch
import utils
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

date = '0907'
lr = 1e-1
epochs = 700
ulb_w = 0.000001
ulb_w2 = 0.001
oe_w = 0.000001
warmup = 10000
warm_mi = 1
lambda_val = 2
un_batch_size = 34
labeled_batchsize = 34
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
mask = list(range(20))
mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
expened_data_num = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
kf = KFold(n_splits=35, shuffle=True, random_state=None)


# 定义训练函数



# 进行十折交叉验证
mse_scores = []
rmse_scores = []
r2_scores = []

total_val_predictions = []
total_val_targets = []

true_data_label = np.load(f'../original_features_labels_{date}_400.npz')
print(true_data_label['features'].shape)
data = true_data_label['features']
lbl = true_data_label['labels']
expended = np.load(f'../features_labels_{date}_400.npz')
print('!!!', expended['features'].shape)
data = np.concatenate((data, expended['features'][0: 0 + expened_data_num, :]))
lbl = np.concatenate((lbl, expended['labels'][0: 0 + expened_data_num]))
true_lbl = lbl[0: 35]
expended_lbl = lbl[35:]
true_data = data[0: 35, :]
expended_data = data[35:, :]
unlabeled_data = np.load(f'../unlabeled_features_{date}_400.npz')['features']

# unlabeled_data = expended_data

for fold, (train_index, test_index) in enumerate(kf.split(true_data)):
    print(f'------------Fold:{fold}  test_index:{test_index}-----------------')
    features_id = mask
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = lbl[train_index], lbl[test_index]
    index1 = np.asarray(list(range(0, expened_data_num, 2)))
    index2 = np.asarray(list(range(1, expened_data_num, 2)))
    X_train1 = X_train[:, features_id]
    y_train1 = y_train
    X_train2 = X_train[:, features_id]
    y_train2 = y_train
    x_test, y_test = data[test_index][:, features_id], lbl[test_index]
    dataset1 = TensorDataset(torch.tensor(X_train1, dtype=torch.float32), torch.tensor(y_train1, dtype=torch.float32))
    dataset2 = TensorDataset(torch.tensor(X_train2, dtype=torch.float32), torch.tensor(y_train2, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader1 = DataLoader(dataset1, drop_last=True, batch_size=labeled_batchsize)
    train_loader2 = DataLoader(dataset2, drop_last=True, batch_size=labeled_batchsize)
    val_loader = DataLoader(val_dataset, drop_last=True, batch_size=1)
    empty_labels = torch.zeros(unlabeled_data.shape[0])
    unlabel_dataset = TensorDataset(torch.tensor(unlabeled_data, dtype=torch.float32), empty_labels)
    unlabel_loader1 = DataLoader(unlabel_dataset, drop_last=True, batch_size=un_batch_size, shuffle=True)
    unlabel_loader2 = DataLoader(unlabel_dataset, drop_last=True, batch_size=un_batch_size, shuffle=True)
    train_batchsize = train_loader1.batch_size
    val_batchsize = val_loader.batch_size

   ############################################## Train Begin########################################################
    model1 = MLP()
    model2 = MLP()



























    
    # model.initialize()
    model1.init_weights()
    model2.init_weights()
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer1 = optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=lr)
    model1.train()
    model2.train()
    model1.to(device)
    model2.to(device)


    # 训练过程
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_mi = 0.0
        running_loss_ulb_0 = 0.0
        running_loss_ulb_1 = 0.0
        mse_loss = 0.0
        sum_mi_loss = 0.0
        sum_oe_loss = 0.0
        ulb_pred_loss = 0.0
        ulb_feat_loss = 0.0
        # 训练循环
        for ((inputs1, targets1), (inputs2, targets2), (inputs_un1, _), (inputs_un2, _)) in zip(train_loader1, train_loader2, unlabel_loader1, unlabel_loader2):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            inputs1, inputs2, inputs_un1, inputs_un2 = inputs1.to(device), inputs2.to(device), inputs_un1.to(device), inputs_un2.to(device)
            targets1, targets2 = targets1.to(device), targets2.to(device)
            # 前向传播
            outputs1, features1 = model1(inputs1)
            outputs2, features2 = model2(inputs2)
            # print(outputs.detach().numpy())
            outputs1_un, features1_un = model1(inputs_un1)
            outputs2_un, features2_un = model2(inputs_un2)
            mi_outputs2_un, mi_features2_un = model2(inputs_un1)

            loss_mse1 = criterion(outputs1, targets1.reshape(-1, 1))
            loss_mse2 = criterion(outputs2, targets2.reshape(-1, 1))



            # print(outputs.detach().numpy())

            loss_oe1 = ordinal_entropy(features1, targets1.reshape(-1, 1))
            loss_oe2 = ordinal_entropy(features2, targets2.reshape(-1, 1))
            if ulb_w > 0:
                # assert False, "Not Done yet"
                loss_ulb_0_unweighted1, ft_rank1, samples1 = ulb_rank(features1_un, lambda_val)
                loss_ulb_0_unweighted2, ft_rank2, samples2 = ulb_rank(features2_un, lambda_val)
                loss_ulb_0_1 = loss_ulb_0_unweighted1 * ulb_w
                loss_ulb_0_2 = loss_ulb_0_unweighted2 * ulb_w
                #print(ft_rank)
                if epoch > warmup:
                    loss_ulb_1_1 = ulb_rank_prdlb(outputs1_un, lambda_val, pred_inp=ft_rank1, samples=samples1) * ulb_w2
                    loss_ulb_1_2 = ulb_rank_prdlb(outputs2_un, lambda_val, pred_inp=ft_rank2, samples=samples2) * ulb_w2
                else:
                    loss_ulb_1_1 = loss_oe1 * 0
                    loss_ulb_1_2 = loss_oe2 * 0
            else:
                loss_ulb_0_1 = loss_oe1 * 0
                loss_ulb_0_2 = loss_oe2 * 0
            if epoch > 0:
                loss_mi = utils.compute_mutual_information(features1_un, mi_features2_un)
            else:
                loss_mi = torch.tensor(0.)

            #loss_mi = loss_mi * min((epoch/epochs) * warm_mi, 1)  # 设置温度超参数
            # loss_mi = loss_mi * 0.001  # 设置温度超参数
            loss_oe1 = loss_oe1 * oe_w
            loss_oe2 = loss_oe2 * oe_w
            loss_mi = loss_mi * 0.0001


            # loss1 = loss_mse1 + loss_oe1 + loss_ulb_0_1 + loss_mi
            # loss2 = loss_mse2 + loss_oe2 + loss_ulb_0_2 + loss_mi
            # loss1 = loss_mse1 + loss_mi
            # loss2 = loss_mse2 + loss_mi
            loss1 = loss_mse1 #+ loss_mi +  loss_ulb_0_1
            loss2 = loss_mse2 #+ loss_mi +  loss_ulb_0_2

            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            # loss1.backward()
            # loss2.backward()
            optimizer1.step()
            optimizer2.step()
            running_loss += (loss1.item() + loss2.item())
            mse_loss += (loss_mse1.item() + loss_mse2.item())
            sum_mi_loss += loss_mi.item()
            sum_oe_loss += (loss_oe1.item() + loss_oe2.item())
            ulb_feat_loss += (loss_ulb_0_1.item() + loss_ulb_0_2.item())
            ulb_pred_loss += (loss_ulb_1_1.item() + loss_ulb_1_2.item())



        # # 更新学习率
        # # scheduler.step()
        # print(  f"Epoch {epoch + 1}/{epochs}, {sum_mi_loss}")
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (len(train_loader1) + len(train_loader2))}, MSE loss:{mse_loss / (len(train_loader1) + len(train_loader2))}, MI_loss:{sum_mi_loss / (len(train_loader1) + len(train_loader2))}, OE_loss:{sum_oe_loss /  (len(train_loader1) + len(train_loader2))}, un feature loss:{ulb_feat_loss /  (len(train_loader1) + len(train_loader2))}, un pred loss:{ulb_pred_loss/  (len(train_loader1) + len(train_loader2))}")
################################################## Train End##########################################
    # 验证过程
    model1.eval()
    model2.eval()
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1, features1 = model1(inputs)
            outputs2, features2 = model2(inputs)
            val_predictions.append(((outputs1.cpu().numpy() + outputs2.cpu().numpy()) / 2))
            val_targets.append(targets.cpu().numpy())

    val_predictions = np.concatenate(val_predictions, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)



    total_val_targets += val_targets.tolist()
    total_val_predictions += val_predictions.tolist()


# 计算验证集上的均方误差 (MSE)
mse = mean_squared_error(total_val_targets, total_val_predictions)
rmse = np.sqrt(mse)
rmse_scores.append(rmse)
mse_scores.append(mse)

# 计算 R2 指标
r2 = r2_score(total_val_targets, total_val_predictions)
r2_scores.append(r2)
fig, ax = plt.subplots()
plt.title("SVR", loc="center")
total_val_targets = np.asarray(total_val_targets)
total_val_predictions = np.asarray(total_val_predictions)
ax.plot(range(total_val_targets.shape[0]), total_val_targets, label='gt')
ax.plot(range(total_val_targets.shape[0]), total_val_predictions, label='pred')
ax.set(xlabel='Sample', ylabel=f'lai')
ax.legend()
plt.show()
torch.save(model1.state_dict(), './model1_weights_0416.pth')
torch.save(model2.state_dict(), './model2_weights_0416.pth')
print(f"十折交叉验证的平均RMSE: {np.mean(rmse_scores)}")
print(f"十折交叉验证的平均R²: {np.mean(r2_scores)}")
# with open(f'./result/result_{epoch}_MI.txt', 'w') as f:
#     f.write(f'MSE:{mse}, R2:{r2}')
#     f.close()