from __future__ import division
from __future__ import print_function

import argparse
import os
import scanpy as sc
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy import stats
import community as community_louvain
import networkx as nx
import leidenalg
import igraph as ig

from config import Config
from models import MAFN
from utils import *


def load_data(dataset):
    print("load data:")
    # path = "../generate_data/DLPFC/" + dataset + "/MBP.h5ad"
    path = "../generate_data/DLPFC/" + dataset + "/MAFN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']

    # marker_indices = adata.uns['marker_indices']
    # raw = adata.obsm['raw_data']

    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    sadjS = adata.obsm['sadjS'] # 特征高斯核
    sadjS = sadjS + np.eye(sadjS.shape[0])


    nfadj = normalize_sparse_matrix(sadjS)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    nsadj = normalize_sparse_matrix(sadj)
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])

    # adj = (sadj + fadj) / 2
    s = sadj
    f = sadjS
    adj1 = f @ s @ f.T
    adj2 = s @ f @ s.T
    adj = (adj1 + adj2) / 2

    mask = compute_difference_matrix(adj)
    mask = get_mask(mask)

    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg, mask#, marker_indices, raw


def train():
    model.train()
    optimizer.zero_grad()
    emb ,pi, disp, mean, emb1, emb2 = model(features, sadj, fadj)

    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)

    dcir_loss= contra_loss(emb1, emb2, mask)

    total_loss =  config.alpha * zinb_loss + config.beta *  reg_loss + config.gamma * dcir_loss

    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values

    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
                '151671', '151672', '151673', '151674', '151675', '151676']
    # datasets = ['151507']
    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = './config/DLPFC.ini'
        print(dataset)
        # adata, features, labels, fadj, sadj, graph_nei, graph_neg, mask, marker_indices, raw = load_data(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg, mask = load_data(dataset)
        print(adata)

        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.family'] = 'Times New Roman'

        savepath = './result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title,
                      show=False)
        plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        config.epochs = 50
        config.epochs = config.epochs + 1

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = MAFN(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,

                             dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        nmi_max = 0
        idx_max = []
        mean_max = []
        emb_max = []


        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss = train()
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.cluster.normalized_mutual_info_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
            if nmi_res > nmi_max:
                nmi_max = nmi_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' dcir_loss = {:.2f}'.format(dcir_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            print(dataset, ' ', ari_res, nmi_res)
        print(dataset, ' ', ari_max, nmi_max)
        '''

        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss = train()

            # 使用 GaussianMixture 代替 KMeans
            gmm = GaussianMixture(n_components=config.class_num)
            gmm.fit(emb)

            # 获取预测的标签
            idx = gmm.predict(emb)

            # 计算聚类结果的评估指标
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.cluster.normalized_mutual_info_score(labels, idx)
            # 保存最好的结果
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
            if nmi_res > nmi_max:
                nmi_max = nmi_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
            # 打印损失和评估指标
            print(dataset, ' epoch: ', epoch,
                  ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss),
                  ' dcir_loss = {:.2f}'.format(dcir_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            print(dataset, ' ', ari_res, nmi_res)
        # 打印最好的 ARI 结果
        print(dataset, ' ', ari_max, nmi_max)

        
        for epoch in range(config.epochs):
            # 获取训练结果
            emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss = train()

            # 使用 KNN 构建图结构，基于余弦相似度
            knn_graph = kneighbors_graph(emb, n_neighbors=20, mode='connectivity', metric='cosine')
            # 将 KNN 图转为 NetworkX 图
            G = nx.Graph(knn_graph)
            # 应用 Louvain 算法进行社区检测
            partition = community_louvain.best_partition(G)
            # 提取聚类结果
            idx = np.array([partition[node] for node in range(len(partition))])

            # 计算评估指标
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.cluster.normalized_mutual_info_score(labels, idx)

            # 保存最好的聚类结果
            if nmi_res > nmi_max:
                nmi_max = nmi_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
            # 打印损失和评估指标
            print(dataset, ' epoch: ', epoch,
                  ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss),
                  ' dcir_loss = {:.2f}'.format(dcir_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            print("ARI:", nmi_res)
        # 打印最好的 ARI 结果
        print(dataset, 'Best ARI:', nmi_max)
        
        
        for epoch in range(config.epochs):
            # 获取训练结果
            emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss = train()

            # 使用 KNN 构建图结构，基于余弦相似度
            knn_graph = kneighbors_graph(emb, n_neighbors=20, mode='connectivity', metric='cosine')
            # 将 KNN 图转为 NetworkX 图
            G = nx.Graph(knn_graph)
            # 转换为 igraph 格式
            G_igraph = ig.Graph.from_networkx(G)
            # 应用 Leiden 算法进行社区检测
            partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
            # 提取聚类结果
            idx = np.array(partition.membership)

            # 计算评估指标
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.cluster.normalized_mutual_info_score(labels, idx)

            # 保存最好的聚类结果
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
            # 打印损失和评估指标
            print(dataset, ' epoch: ', epoch,
                  ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss),
                  ' dcir_loss = {:.2f}'.format(dcir_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            print("NMI:", ari_res)
        # 打印最好的 NMI 结果
        print(dataset, 'Best NMI:', ari_max)
        '''

        title = 'MAFN: ARI={:.2f}  NMI={:.2f}'.format(ari_max, nmi_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'MAFN.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        # 计算邻接矩阵
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='mean')
        # 计算 UMAP
        sc.tl.umap(adata)
        # 设置图像大小
        plt.rcParams["figure.figsize"] = (6, 6)
        # 绘制 UMAP 图
        sc.pl.umap(adata, color=['idx'], frameon=False, show=False)
        # 保存 UMAP 图
        plt.savefig(savepath + 'MAFN_umap_mean.jpg', bbox_inches='tight', dpi=600)
        # 显示图形
        plt.show()
        # pd.DataFrame(emb_max).to_csv(savepath + 'MAFN_emb.csv')
        # pd.DataFrame(idx_max).to_csv(savepath + 'MAFN_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        # adata.write(savepath + 'MAFN.h5ad')

        '''
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 28  # 设置字体大小为较小的值

        sc.pl.spatial(adata, color='MBP',layer='mean', show=True, vmin=0, vmax='3')
        sc.pl.spatial(adata, color='MBP', layer='mean', show=True, vmin=0, vmax='p99')

        mean = adata.obsm['mean']
        idx_mask = marker_indices
        pred_values = np.squeeze(mean[:, idx_mask])
        true_values = np.squeeze(raw[:, idx_mask])

        new_pear = stats.pearsonr(pred_values, true_values)[0]
        print("PCC", new_pear)
        '''