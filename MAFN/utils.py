import scipy.sparse as sp
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
EPS = 1e-15


def regularization_loss(emb, graph_nei, graph_neg):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cross_correlation(Z_v1, Z_v2):
    """
    calculate the cross-view correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    """
    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())

def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()

def dicr_loss(com1, com2):
    """
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        AZ: the propagated fusion embedding AZ
        Z: the fusion embedding Z
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    S_F_ae = cross_correlation(com1.t(), com2.t())

    # loss of FCR
    L_F_ae = correlation_reduction_loss(S_F_ae)

    # loss of DICR
    loss_dicr =  L_F_ae

    return loss_dicr


def spatial_construct_graph1(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]]=1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    # sadj = (sadj + sadj.T) / 2
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg#, nsadj


def spatial_construct_graph(positions, k=15):
    print("start spatial construct graph")
    A = euclidean_distances(positions)
    tmp = 0
    mink = 2
    for t in range(100, 1000, 100):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 100, 1000, 10):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 10, 1000, 5):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            A = A1
            break
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/edge.csv', index, delimiter=',')

    graph_nei = torch.from_numpy(A)
    # print(type(graph_nei),graph_nei)
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    # sadj = (sadj + sadj.T) / 2
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg#, nsadj


def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k + 1, mode=mode,metric=metric,include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/fadj.csv', index, delimiter=',')
    fadj = sp.coo_matrix(A, dtype=np.float32)
    # fadj = (fadj + fadj.T) / 2
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj#, nfadj


def get_adj(data, pca=None, k=25, mode="connectivity", metric="cosine"):
    if pca is not None:
        data = dopca(data, dim=pca)
        data = data.reshape(-1, 1)
    A = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    # S = cosine_similarity(data)
    return adj, adj_n  # , S


def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([np.mean(data[labels == i], axis=0) for i in range(n_clusters)])


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # 如果 sparse_mx 是 numpy.matrix 类型，先将其转换为 numpy.ndarray
    if isinstance(sparse_mx, np.matrix):
        sparse_mx = np.array(sparse_mx)
    # 确保输入是稀疏矩阵，如果是 dense 矩阵，先转换为稀疏矩阵
    if isinstance(sparse_mx, np.ndarray):
        sparse_mx = coo_matrix(sparse_mx)  # 转换为稀疏矩阵
    # 将稀疏矩阵转换为 COO 格式
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 获取稀疏矩阵的行、列索引
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 获取稀疏矩阵的值
    values = torch.from_numpy(sparse_mx.data)
    # 获取稀疏矩阵的形状
    shape = torch.Size(sparse_mx.shape)

    # 返回 Torch 稀疏张量
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    mx = norm_adj(mx)

    return mx


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


class Colors():
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#633194",
        "#8c564b",
        "#F73BAD",
        "#ad494a",
        "#F6E800",
        "#01F7F7",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#8c6d31"]

'''new'''

def compute_difference_matrix(A):  # 差异矩阵
    """
        计算模块度矩阵 B
        参数：
        A -- 邻接矩阵（num_nodes x num_nodes）
        返回：
        B -- 模块度矩阵（num_nodes x num_nodes）
    """
    # 计算节点的度数向量 degree
    degree = A.sum(axis=1)
    # 计算边的总数 m
    m = degree.sum() / 2
    # 计算每个元素的模块度矩阵B的值
    B = A - np.outer(degree, degree) / (2 * m)
    # 如果计算结果为 NaN 或 inf，可以进行修复
    # 将 NaN 或 inf 替换为 0
    # B[np.isnan(B) | np.isinf(B)] = 0

    return B


def get_mask(matrix):
    """
    输入一个矩阵，返回一个与之相同形状的矩阵，元素大于0的位置为1，小于0的位置为0。
    对角线上的元素强制为1。

    Parameters:
    matrix (np.ndarray): 输入的矩阵

    Returns:
    np.ndarray: 输出的矩阵，元素大于0为1，小于0为0
    """
    mask = (matrix > 0).astype(int)  # 将布尔值转换为整数（True变为1，False变为0）
    # 强制对角线上的元素为1
    np.fill_diagonal(mask, 1)

    return mask


class pseudo_positive_contrastive_loss(torch.nn.Module):
    def __init__(self, tau):
        super(pseudo_positive_contrastive_loss, self).__init__()
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):  # 余弦相似度（归一化的内积）
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, Q: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        # refl_sim = f(self.sim(z1, z1))  # 自身相似度矩阵
        between_sim = f(self.sim(z1, z2))  # 两组特征之间的相似度矩阵

        # 根据Q矩阵筛选正样本
        Q = torch.tensor(Q)
        pos_sim = Q * between_sim  # 只保留Q矩阵为1的相似度值
        pos_sim_sum = pos_sim.sum(dim=1)  # 每个样本的正样本相似度和,形状(N,)

        # 计算负样本的相似度和（排除自身相似度）
        # neg_sim_sum = refl_sim.sum(dim=1) + between_sim.sum(dim=1) - refl_sim.diag() - pos_sim_sum
        neg_sim_sum = between_sim.sum(dim=1) - pos_sim_sum

        # 计算损失
        loss = - torch.log(pos_sim_sum / (pos_sim_sum + neg_sim_sum))
        # loss = - torch.log(pos_sim_sum / neg_sim_sum)
        return loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, Q: torch.Tensor, mean: bool = True):
        l = self.semi_loss(z1, z2, Q)
        ret = l
        ret = ret.mean() if mean else ret.sum()
        return ret


def contra_loss(emb1, emb2, Q):
    conloss = pseudo_positive_contrastive_loss(0.5)  # tau
    loss = conloss(emb1, emb2, Q)

    return loss


def threshold_matrix(matrix):
    # 使用布尔索引将大于 0 的元素设为 1，其他设为 0
    return np.where(matrix > 0, 1, 0)


def gaussian_kernel_matrix(X):

    # 计算样本间的欧氏距离矩阵 (n, n)
    X = torch.tensor(X)
    X_squared = torch.sum(X ** 2, dim=1).view(-1, 1)  # (n, 1)
    dist_squared = X_squared + X_squared.T - 2 * torch.matmul(X, X.T)  # (n, n)
    # 计算每对样本之间的欧氏距离（包括自己）
    neighbors_mean = dist_squared.mean(dim=1).view(-1, 1)  # (n, 1)
    dist_squared_with_neighbors = (neighbors_mean + neighbors_mean.T + dist_squared) / 3  # (n, n)
    # 计算高斯核矩阵
    K = torch.exp(-dist_squared / dist_squared_with_neighbors)
    # 保证对称
    K = (K + K.T) / 2

    return K


def local_affinity(W, k=14):

    # 将 W 转换为 numpy 数组以便使用 KNN 算法
    W_np = W.numpy()
    W_np = np.nan_to_num(W_np, nan=0)
    # 计算每一行的邻居
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(W_np)
    # 获取每个节点的 k 个最近邻的索引
    _, indices = nn.kneighbors(W_np)  # 返回每个点的 k 个最近邻的索引
    # 初始化局部相关矩阵 S
    S = torch.zeros_like(W)
    # 获取邻居的索引（排除自身）
    indices = indices[:, 1:]  # 每个点的第一个邻居是它自己，排除它
    # 获取每个节点邻居权重的总和
    row_sums = W.gather(1, torch.tensor(indices))
    # 归一化每个邻居的权重
    S.scatter_(1, torch.tensor(indices), row_sums)
    S = (S + S.T) / 2

    return S


def features_construct_graph4(features, pca=None, k=14):  # 特征模态矩阵S
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)

    A = gaussian_kernel_matrix(features)
    A = local_affinity(A, k=k)

    fadj = sp.coo_matrix(A, dtype=np.float32)
    # fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj


def compute_difference_matrix(A):  # 差异矩阵

    # 计算节点的度数向量 degree
    degree = A.sum(axis=1)
    # 计算边的总数 m
    m = degree.sum() / 2
    # 计算每个元素的模块度矩阵B的值
    B = A - np.outer(degree, degree) / (2 * m)

    return B


def get_mask(matrix):
    """
    输入一个矩阵，返回一个与之相同形状的矩阵，元素大于0的位置为1，小于0的位置为0。
    对角线上的元素强制为1。

    Parameters:
    matrix (np.ndarray): 输入的矩阵

    Returns:
    np.ndarray: 输出的矩阵，元素大于0为1，小于0为0
    """
    mask = (matrix > 0).astype(int)  # 将布尔值转换为整数（True变为1，False变为0）
    # 强制对角线上的元素为1
    np.fill_diagonal(mask, 1)

    return mask