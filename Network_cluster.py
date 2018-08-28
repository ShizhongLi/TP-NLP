# coding: utf-8

import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pylab as plt
import copy
from sklearn.cluster import SpectralClustering
from sklearn import metrics
# from Cluster_score_Userdefine import Calinski_harabaz_score_cos
import pandas as pd 
from pylab import mpl

# ———————————————------— 层次聚类 ———————————————------—



# 用于去掉合并的结点，只保留原始节点
def valid_k_cluster(Matrix, k_cluster):
    temp = []
    for num in list(set(k_cluster)):
        if num <= Matrix.shape[0] - 1:
            temp.append(num)
    return temp

# 用于获取聚类对应的分组结果
def k_cluster_cols(Matrix, k_cluster):
    temp = []
    for i in k_cluster:
        
        col_name = list(Matrix.index)[int(i)]
        temp.append(col_name)
    return temp


# 获取类别具体内容
def Clustering_content(Matrix, cutoff, Z):
    Z_new = np.zeros([len(Z), 3])
    for i in range(len(Z_new)):
        Z_new[i][0] = Z[i][0]
        Z_new[i][1] = Z[i][1]
        Z_new[i][2] = Matrix.shape[0] + i  # 计算每次合并后的节点的序号


    Z_dist = {}
    temp = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4= []
    cluster5 = []
    cluster6 = []
    
    cluster1_temp = []
    cluster2_temp=[]
    cluster3_temp=[]
    cluster4_temp=[]
    cluster5_temp=[]
    cluster6_temp=[]
    
   
    for i in range(len(Z_new)-1,-1,-1):
        if Z[i][2] <= cutoff:
            for j in range(len(Z_new)-1,-1,-1):
                if i != j and Z[j][2] <= cutoff:
                    if [x for x in Z_new[i] if x in Z_new[j]] != []:  # 如果Z_new[i]和Z_new[j]中的元素有交集
                        
                        # 如果i行或j行的结点有一个在cluster1_temp里面的话，那么i和j两行都归到cluster1_temp
                        if cluster1_temp==[] or([x for x in Z_new[i] if x in cluster1_temp] != [] or [x for x in Z_new[j] if x in cluster1_temp] != []):
                            for k in range(3):
                                cluster1.append(list(Z_new[i])[k])
                                cluster1.append(list(Z_new[j])[k])
                            cluster1_temp = list(set(cluster1))

                        elif cluster2_temp==[] or ([x for x in Z_new[i] if x in cluster2_temp] != [] or [x for x in Z_new[j] if x in cluster2_temp] != []):
                            for k in range(3):
                                cluster2.append(list(Z_new[i])[k])
                                cluster2.append(list(Z_new[j])[k])
                            cluster2_temp = list(set(cluster2))

                        elif cluster3_temp==[] or ([x for x in Z_new[i] if x in cluster3_temp] != [] or [x for x in Z_new[j] if x in cluster3_temp] != []):
                            for k in range(3):
                                cluster3.append(list(Z_new[i])[k])
                                cluster3.append(list(Z_new[j])[k])
                            cluster3_temp = list(set(cluster3))
                        elif cluster4_temp==[] or ([x for x in Z_new[i] if x in cluster4_temp] != [] or [x for x in Z_new[j] if x in cluster4_temp] != []):
                            for k in range(3):
                                cluster4.append(list(Z_new[i])[k])
                                cluster4.append(list(Z_new[j])[k])
                            cluster4_temp = list(set(cluster4))
                        elif cluster5_temp==[] or ([x for x in Z_new[i] if x in cluster5_temp] != [] or [x for x in Z_new[j] if x in cluster5_temp] != []):
                            for k in range(3):
                                cluster5.append(list(Z_new[i])[k])
                                cluster5.append(list(Z_new[j])[k])
                            cluster5_temp = list(set(cluster5))
                        elif cluster6_temp==[] or ([x for x in Z_new[i] if x in cluster6_temp] != [] or [x for x in Z_new[j] if x in cluster6_temp] != []):
                            for k in range(3):
                                cluster6.append(list(Z_new[i])[k])
                                cluster6.append(list(Z_new[j])[k])
                            cluster6_temp = list(set(cluster6))
            
             # 如果i行或j行的结点有一个在cluster1_temp里面的话，那么i和j两行都归到cluster1_temp
            if cluster1_temp==[] or [x for x in Z_new[i] if x in cluster1_temp] != [] :
                for k in range(3):
                    cluster1.append(list(Z_new[i])[k])                
                cluster1_temp = list(set(cluster1))

            elif cluster2_temp==[] or [x for x in Z_new[i] if x in cluster2_temp] != [] :
                for k in range(3):
                    cluster2.append(list(Z_new[i])[k])                
                cluster2_temp = list(set(cluster2))

            elif cluster3_temp==[] or [x for x in Z_new[i] if x in cluster3_temp] != [] :
                for k in range(3):
                    cluster3.append(list(Z_new[i])[k])                  
                cluster3_temp = list(set(cluster3))
            elif cluster4_temp==[] or [x for x in Z_new[i] if x in cluster4_temp] != [] :
                for k in range(3):
                    cluster4.append(list(Z_new[i])[k])                    
                cluster4_temp = list(set(cluster4))
            elif cluster5_temp==[] or [x for x in Z_new[i] if x in cluster5_temp] != [] :
                for k in range(3):
                    cluster5.append(list(Z_new[i])[k])                   
                cluster5_temp = list(set(cluster5))
            elif cluster6_temp==[] or [x for x in Z_new[i] if x in cluster6_temp] != [] :
                for k in range(3):
                    cluster6.append(list(Z_new[i])[k])                   
                cluster6_temp = list(set(cluster6))
   

    # 聚类数计数并只保留非空的类别                 
    n_cluster=0  # 聚类数计数       
    cluster_list=[cluster1_temp,cluster2_temp,cluster3_temp,cluster4_temp,cluster5_temp,cluster6_temp]
    
    k_cluster=[] # 聚类结果：变量名形式
    k_cluster2=[] # 聚类结果：数值形式
    for clu in cluster_list:
        if clu!=[]:
            n_cluster=n_cluster+1  
            k_cluster.append(k_cluster_cols(Matrix, valid_k_cluster(Matrix, clu)))
            #k_cluster_cols(Matrix, valid_k_cluster(Matrix, clu)) # 测试
            k_cluster2.append(valid_k_cluster(Matrix, clu))
    label_list,name_list=Clustering_labels(Matrix, k_cluster)
    
    #print('k_cluster',k_cluster,'label_list',label_list)
    
    return k_cluster,label_list


# 获取类别具体内容
def Clustering_labels(Matrix, k_cluster):
    label_list=np.zeros([len(list(Matrix.index)),])
    name_list=list(Matrix.index)
    for k in range(len(list(Matrix.index))):
        for i in range(len(k_cluster)):
            for j in range(len(k_cluster[i])):
            
                if k_cluster[i][j]==list(Matrix.index)[k]:
                    label_list[k]=i+1
                    
            
    return label_list,name_list


# 层次聚类-主调用函数
def Net_Hierarchical_clustering(DF,cutoff = 0.5,metric_name='jaccard',figsize_v=(10, 5)):  ## 输入邻接矩阵
    '''
    函数用于层次聚类，并输出聚类结果, 计算聚类得分
    Input :
            DF : 数据框（一行为一个对象进行聚类，行坐标为聚类对象）
            cutoff     ：层次聚类的分割点, 默认cutoff = 0.5
            metric_name: 距离计算, 默认metric_name='jaccard'
            figsize_v:图像大小
    Output:
            k_cluster  ：具体的聚类结果
            label_list ：聚类标签
    '''
    cutoff = cutoff  # cutoff是阈值
    # 层次聚类:
    Z = sch.linkage(DF, method='average') # 层次聚类，平均连接方式
    plt.figure(figsize=figsize_v)
    plt.savefig('plot_dendrogram.png')
    Z2 = sch.dendrogram(Z, color_threshold=cutoff,labels=list(DF.index)) # 绘制层次聚类图
    mpl.rcParams['font.sans-serif'] = ['FangSong']    # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False        # 解决保存图像是负号'-'显示为方块的问题　
    plt.show()
    # 获取每个类别的具体内容
    k_cluster,label_list = Clustering_content(DF, cutoff, Z)
    
    #if len(k_cluster) == 1:
     #   print('只有一个类别, 无法计算calinski_harabaz_score和silhouette_score')
    #else: 
        # Calinski-Harabasz
        # 评分类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。
        # score = metrics.calinski_harabaz_score(DF, label_list)

        # 轮廓系数评分
        # silhouette_avg = metrics.silhouette_score(DF, label_list, metric = metric_name)  
        # 轮廓系数(silhouette coefficient) 结合了凝聚度和分离度，其计算步骤如下：
        # 对于第 i 个对象，计算它到所属簇中所有其他对象的平均距离，记 ai （体现凝聚度）
        # 对于第 i 个对象和不包含该对象的任意簇，计算该对象到给定簇中所有对象的平均距离，记 bi （体现分离度）
        # 第 i 个对象的轮廓系数为 si = (bi-ai)/max(ai, bi)  //回头研究一下 wordpress 的公式插件去
        # 从上面可以看出，轮廓系数取值为[-1, 1]，其值越大越好，且当值为负时，表明 ai<bi，样本被分配到错误的簇中，聚类结果不可接受。
        # 对于接近0的结果，则表明聚类结果有重叠的情况。
        #print('calinski_harabaz_score',score,'silhouette_avg',silhouette_avg)
    return k_cluster,label_list

# ——————————————----—-----—- 层次聚类 —————————————----—------—



# ——————————————----—------— 谱聚类 ——————————————----—------—

def Self_spectral_clustering(Matrix,nclusters=3, affinity_v='rbf'):  ## 输入参数为邻接矩阵
    ADJ_Spectral_clustering = copy.deepcopy(Matrix)
    # 网格搜索调节参数
    max_score = 0
    max_score_nclusters = 0
    for index, gamma in enumerate((0.01, 0.1, 1, 10)):
        for index, k in enumerate((3, 4, 5)):
            # 输入的数据默认应该用原始数据，如果要用邻接矩阵作为输入数据，那么就应该设置 affinity='precomputed'
            y_pred = SpectralClustering(n_clusters=k, affinity=affinity_v).fit_predict(
                ADJ_Spectral_clustering)
            score = metrics.calinski_harabaz_score(ADJ_Spectral_clustering, y_pred)
            if score > max_score:
                max_score = score
                max_score_nclusters = k
            # Calinski-Harabasz Score 越小，相当于组间距越不明显，分组效果越差

   
    
    if nclusters==0: # 没有指定聚类数的情况，按照最优的参数拟合
        n_clusters=max_score_nclusters
    else:  # 如果有指定聚类数，按照聚类数拟合
        n_clusters=nclusters
        
    y_pred = SpectralClustering(n_clusters=n_clusters).fit_predict(
        ADJ_Spectral_clustering)
    
    
    # Calinski-Harabasz
    # 评分类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。
    #CH_score = metrics.calinski_harabaz_score(ADJ_Spectral_clustering, y_pred)
    
    #CH_score_liu=Calinski_harabaz_score_cos(ADJ_Spectral_clustering.as_matrix(columns=None), y_pred)
    
    # 轮廓系数评分
    #silhouette_avg = metrics.silhouette_score(ADJ_Spectral_clustering, y_pred, metric = 'jaccard')  
    # 轮廓系数(silhouette coefficient) 结合了凝聚度和分离度，其计算步骤如下：
    # 对于第 i 个对象，计算它到所属簇中所有其他对象的平均距离，记 ai （体现凝聚度）
    # 对于第 i 个对象和不包含该对象的任意簇，计算该对象到给定簇中所有对象的平均距离，记 bi （体现分离度）
    # 第 i 个对象的轮廓系数为 si = (bi-ai)/max(ai, bi)  //回头研究一下 wordpress 的公式插件去
    # 从上面可以看出，轮廓系数取值为[-1, 1]，其值越大越好，且当值为负时，表明 ai<bi，样本被分配到错误的簇中，聚类结果不可接受。
    # 对于接近0的结果，则表明聚类结果有重叠的情况。
    
    ADJ_Spectral_clustering['聚类类别'] = y_pred
   
    return ADJ_Spectral_clustering,n_clusters,y_pred


def Net_Spectral_clustering(data,nclusters=3,affinity_v='rbf'):  ## 输入邻接矩阵
    '''
    函数用于谱聚类，并输出聚类结果, 计算聚类得分
    Input :
            data : 邻接矩阵或m×n的矩阵, 其中m为样本数, n为样本特征数(均为DataFrame)
            nclusters  ：分类类别数, 默认nclusters=3
            affinity_v : 邻接矩阵计算方式, 默认值为'rbf'。若输入的data为邻接矩阵, 则affinity_v应设为'precomputed'。
    Output:
            k_cluster  ：具体的聚类结果
            y_pred:聚类标签
    '''
    # 谱聚类
    nclusters=nclusters
    ADJ_Spectral_clustering, max_score_nclusters,y_pred = Self_spectral_clustering(data,nclusters,affinity_v)
    
    # 获取每个类别的具体内容,
    k_cluster = [[]] * max_score_nclusters
    for i in range(max_score_nclusters):
        k_cluster[i] = list(ADJ_Spectral_clustering[ADJ_Spectral_clustering['聚类类别'] == i].index)

    return k_cluster,y_pred

# ——————————————---—-------— 谱聚类 ——————————————----—------—


# ———————————————------— Fast_Unfolding ———————————————------—

# 整理数据格式
class data_deal:
    def __init__(self, X):
        self.temp1, self.temp11 = self.get_node_info(X)

    # 整理邻接矩阵到网络图的绘图格式
    def get_node_info(self, X):
        temp1 = []
        temp11 = []
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    temp0 = [str(i), str(j), str(X.iloc[i, j])]
                    temp1.append(temp0)
                    temp00 = [X.columns[i], X.index[j], X.iloc[i, j]]
                    temp11.append(temp00)
        return [temp1, temp11]


# 导入数据
def loadData(temp11):
    vector_dict = {}
    edge_dict = {}

    for line in temp11:
        lines = line
        for i in range(2):
            if lines[i] not in vector_dict:
                # put the vector into the vector_dict
                vector_dict[lines[i]] = True
                # put the edges into the edge_dict
                edge_list = []
                if len(lines) == 3:  ##输入数据中包含权重
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:  ##输入数据中不含权重，默认给权重1
                    edge_list.append(lines[1 - i] + ":" + "1")
                edge_dict[lines[i]] = edge_list
            else:
                edge_list = edge_dict[lines[i]]
                if len(lines) == 3:
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:
                    edge_list.append(lines[1 - i] + ":" + "1")
                edge_dict[lines[i]] = edge_list
    return [vector_dict, edge_dict]

# 计算当前模型的Q值（模块度）
def modularity(vector_dict, edge_dict):
    Q = 0.0
    # m represents the total wight
    m = 0

    # 对edge_dict中的值进行遍历。即对有连边的节点序号进行遍历。i为有连边的节点的序号。
    for i in edge_dict.keys():
        # 把节点i对应的边列表取出
        edge_list = edge_dict[i]

        # 对节点i下的边列表进行遍历
        for j in range(len(edge_list)):
            # 按：符号将边和边的权重拆分为独立元素
            l = edge_list[j].strip().split(":")

            # m是节点i的所有连边的权重和。即节点i的度
            m += float(l[1].strip())

            # cal community of every vector
    # 获取这个集群中的节点集合
    # find member in every community
    community_dict = {}

    # 对vector_dict进行值遍历
    for i in vector_dict.keys():

        # 如果节点i不存在community_dict中
        if vector_dict[i] not in community_dict:
            community_list = []
            # print(community_list)

        # 如果节点i存在community_dict中，则把community_dict对应节点i的值提取出来放在community_list里面。因为要保留这个节点的list
        else:
            community_list = community_dict[vector_dict[i]]

        # community_list 中增加i节点
        community_list.append(i)

        # 为community_dict新建一个值节点i（即 vector_dict[i]），并将community_list作为节点i的属性值
        community_dict[vector_dict[i]] = community_list

    # cal inner link num and degree
    innerLink_dict = {}
    for i in community_dict.keys():
        sum_in = 0.0
        sum_tot = 0.0
        # vector num
        # 获取集群中的第i节点
        vector_list = community_dict[i]
        # print "vector_list : ", vector_list
        # two loop cal inner link
        if len(vector_list) == 1:
            tmp_list = edge_dict[vector_list[0]]
            tmp_dict = {}
            for link_mem in tmp_list:
                l = link_mem.strip().split(":")
                tmp_dict[l[0]] = l[1]
            if vector_list[0] in tmp_dict:
                sum_in = float(tmp_dict[vector_list[0]])
            else:
                sum_in = 0.0
        else:
            for j in range(0, len(vector_list)):
                link_list = edge_dict[vector_list[j]]
                tmp_dict = {}
                for link_mem in link_list:
                    l = link_mem.strip().split(":")
                    # split the vector and weight
                    tmp_dict[l[0]] = l[1]
                for k in range(0, len(vector_list)):
                    if vector_list[k] in tmp_dict:
                        sum_in += float(tmp_dict[vector_list[k]])

        # cal degree
        for vec in vector_list:
            link_list = edge_dict[vec]
            for i in link_list:
                l = i.strip().split(":")
                sum_tot += float(l[1])
        Q += ((sum_in / m) - (sum_tot / m) * (sum_tot / m))
        ##模块度的计算公式见 https://blog.csdn.net/marywbrown/article/details/62059231 中的公式（5）
    return Q


# 把某个节点移动到相邻集群时，比较移动前后Q值是否增加。
def chage_community(vector_dict, edge_dict, Q):
    vector_tmp_dict = {}
    for key in vector_dict:
        vector_tmp_dict[key] = vector_dict[key]

    # 遍历所有节点
    for key in vector_tmp_dict.keys():

        # edge_dict[key]的格式为：['0:1', '0:1', '1:1', '2:1', '4:1', '13:1']
        # neighbor_vector_list是节点key对应的边的信息
        neighbor_vector_list = edge_dict[key]

        # 遍历与该节点相连的所有节点。
        for vec in neighbor_vector_list:
            # 提取节点序号
            ori_com = vector_tmp_dict[key]
            # 提取节点某个连边的信息，并拆分为边和权重两个元素
            vec_v = vec.strip().split(":")

            # compare the list_member with ori_com
            # 如果当前节点不等于他的当前相邻节点vec_v。那么把当前节点的值改为他的当前相邻节点vec_v的值。
            if ori_com != vector_tmp_dict[vec_v[0]]:
                vector_tmp_dict[key] = vector_tmp_dict[vec_v[0]]
                Q_new = modularity(vector_tmp_dict, edge_dict)
                # print Q_new
                # 如果更改后新的Q值大于原来的，那么保留这个Q值以及和这个节点的更改
                if (Q_new - Q) > 0:
                    Q = Q_new
                # 如果更改后新的Q值不大于原来的，那么把改回原本的节点
                else:
                    vector_tmp_dict[key] = ori_com
    return vector_tmp_dict, Q


def modify_community(vector_dict):
    # modify the community
    community_dict = {}
    community_num = 0
    for community_values in vector_dict.values():
        if community_values not in community_dict:
            community_dict[community_values] = community_num
            community_num += 1
    for key in vector_dict.keys():
        vector_dict[key] = community_dict[vector_dict[key]]
    return community_num


def rebuild_graph(vector_dict, edge_dict, community_num):
    vector_new_dict = {}
    edge_new_dict = {}
    # cal the inner connection in every community
    community_dict = {}
    for key in vector_dict.keys():
        if vector_dict[key] not in community_dict:
            community_list = []
        else:
            community_list = community_dict[vector_dict[key]]

        community_list.append(key)
        community_dict[vector_dict[key]] = community_list

    # cal vector_new_dict
    for key in community_dict.keys():
        vector_new_dict[str(key)] = str(key)

    # put the community_list into vector_new_dict

    # 计算内部连线
    # cal inner link num
    innerLink_dict = {}
    for i in community_dict.keys():
        sum_in = 0.0
        # vector num
        vector_list = community_dict[i]
        # two loop cal inner link
        if len(vector_list) == 1:
            sum_in = 0.0
        else:
            for j in range(0, len(vector_list)):
                link_list = edge_dict[vector_list[j]]
                tmp_dict = {}
                for link_mem in link_list:
                    l = link_mem.strip().split(":")
                    # split the vector and weight
                    tmp_dict[l[0]] = l[1]
                for k in range(0, len(vector_list)):
                    if vector_list[k] in tmp_dict:
                        sum_in += float(tmp_dict[vector_list[k]])

        inner_list = []
        inner_list.append(str(i) + ":" + str(sum_in))
        edge_new_dict[str(i)] = inner_list

    # cal outer link num
    community_list = list(community_dict.keys())
    for i in range(len(community_list)):
        for j in range(len(community_list)):
            if i != j:
                sum_outer = 0.0
                member_list_1 = community_dict[community_list[i]]
                member_list_2 = community_dict[community_list[j]]

                for i_1 in range(len(member_list_1)):
                    tmp_dict = {}
                    tmp_list = edge_dict[member_list_1[i_1]]

                    for k in range(len(tmp_list)):
                        tmp = tmp_list[k].strip().split(":");
                        tmp_dict[tmp[0]] = tmp[1]
                    for j_1 in range(len(member_list_2)):
                        if member_list_2[j_1] in tmp_dict:
                            sum_outer += float(tmp_dict[member_list_2[j_1]])

                if sum_outer != 0:
                    inner_list = edge_new_dict[str(community_list[i])]
                    inner_list.append(str(j) + ":" + str(sum_outer))
                    ## 网上建议，上一语句改为以下语句：（目前跑了数据，两句跑出的结果一样）
                    #                     inner_list.append(str(community_list[j]) + ":" + str(sum_outer))
                    #                     print('对比看看j和community_list[j]', j, community_list[j])
                    edge_new_dict[str(community_list[i])] = inner_list
    return vector_new_dict, edge_new_dict, community_dict


def fast_unfolding(vector_dict, edge_dict):
    # 1. initilization:put every vector into different communities
    #   the easiest way:use the vector num as the community num
    for i in vector_dict.keys():
        vector_dict[i] = i

    # print "vector_dict : ", vector_dict
    # print "edge_dict : ", edge_dict
    # 计算Q值
    Q = modularity(vector_dict, edge_dict)

    # 2. for every vector, chose the community
    Q_new = 0.0
    while (Q_new != Q):
        Q_new = Q
        # 移动某个节点看看Q值是否有变大，有的话保留Q值和节点移动后的集群。如果没有变大则返回原本的Q值和移动前的节点集群
        vector_dict, Q = chage_community(vector_dict, edge_dict, Q)

    # 计算集群节点数
    community_num = modify_community(vector_dict)
    # print ("Q = ", Q)
    # print ("vector_dict.key : ", vector_dict.keys())
    # print ("vector_dict.value : ", vector_dict.values())
    Q_best = Q
    while (True):
        # 3. rebulid new graph, re_run the second step
        # print ("edge_dict : ",edge_dict)
        # print ("vector_dict : ",vector_dict)
        # print ("\n rebuild")
        vector_dict, edge_new_dict, community_dict = rebuild_graph(vector_dict, edge_dict, community_num)
        # print vector_dict
        # print ("community_dict : ", community_dict)

        Q_new = 0.0
        while (Q_new != Q):
            Q_new = Q
            vector_dict, Q = chage_community(vector_dict, edge_new_dict, Q)
        community_num = modify_community(vector_dict)
        # print ("Q = ", Q)
        if (Q_best == Q):
            break
        Q_best = Q
        vector_result = {}
        for key in community_dict.keys():
            value_of_vector = community_dict[key]
            for i in range(len(value_of_vector)):
                vector_result[value_of_vector[i]] = str(vector_dict[str(key)])
        for key in vector_result.keys():
            vector_dict[key] = vector_result[key]
        # print ("vector_dict.key : ", vector_dict.keys())
        # print ("vector_dict.value : ", vector_dict.values())

    # 获得最终结果
    # get the final result
    vector_result = {}
    for key in community_dict.keys():
        value_of_vector = community_dict[key]
        for i in range(len(value_of_vector)):
            vector_result[value_of_vector[i]] = str(vector_dict[str(key)])
    for key in vector_result.keys():
        vector_dict[key] = vector_result[key]

    return [Q_best, community_dict, vector_dict.keys(), vector_dict.values()]

# Fast_unfolding 主调用函数
def Net_Fast_Unfolding(ADJ_Matrix):  ## 输入邻接矩阵
    '''
    函数用于谱聚类，并输出聚类结果, 计算聚类得分
    Input :
            ADJ_Matrix    : 邻接矩阵DataFrame
    Output:
            k_cluster     ：具体的聚类结果
            community_list: 聚类类别
            Q_best:模块度
            community_dict：聚类结果
    '''
    # 整理数据
    data = data_deal(ADJ_Matrix)
    vector_dict, edge_dict = loadData(data.temp1)
    # Fast_Unfolding聚类
    Q_best, community_dict, vector_dict_keys, vector_dict_values = fast_unfolding(vector_dict, edge_dict)
    #print('Q值：', Q_best, '分类结果：', community_dict)
    # 获取每个类别的具体内容
    k_cluster = []
    for k in community_dict.keys():
        k_cluster.append([ADJ_Matrix.columns[int(x)] for x in community_dict[k]])
        
    y_pred = np.zeros(ADJ_Matrix.shape[0])
    num = 0
    for val in community_dict.values():
        for j in val:
            y_pred[int(j)] = num
        num = num + 1
    
    #if len(k_cluster) == 1:
    #    print('只有一个类别, 无法计算calinski_harabaz_score和silhouette_score')
    #else: 
        # Calinski-Harabasz
        # 评分类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。
        
        #CH_score = metrics.calinski_harabaz_score(ADJ_Matrix, y_pred)

        #CH_score_liu=Calinski_harabaz_score_cos(ADJ_Matrix.as_matrix(columns=None), y_pred)
        # 轮廓系数评分
        #silhouette_avg = metrics.silhouette_score(ADJ_Matrix, y_pred, metric = 'jaccard')  
        # 轮廓系数(silhouette coefficient) 结合了凝聚度和分离度，其计算步骤如下：
        # 对于第 i 个对象，计算它到所属簇中所有其他对象的平均距离，记 ai （体现凝聚度）
        # 对于第 i 个对象和不包含该对象的任意簇，计算该对象到给定簇中所有对象的平均距离，记 bi （体现分离度）
        # 第 i 个对象的轮廓系数为 si = (bi-ai)/max(ai, bi)  /研究一下 wordpress 的公式插件去
        # 从上面可以看出，轮廓系数取值为[-1, 1]，其值越大越好，且当值为负时，表明 ai<bi，样本被分配到错误的簇中，聚类结果不可接受。
        # 对于接近0的结果，则表明聚类结果有重叠的情况。

        #print( 'CH_score:', CH_score,'CH_score_liu',CH_score_liu,'silhouette_avg',silhouette_avg)
    
    community_list = list(community_dict)
    
    return k_cluster, community_list,Q_best,community_dict

# ———————————————------— Fast_Unfolding ———————————————------—


if __name__ == "__main__":
    ADJ_Matrix = pd.read_excel('./data/df_copd_ADJ.xlsx')
    Net_Hierarchical_clustering(ADJ_Matrix,cutoff = 0.3,metric_name='jaccard')
    Net_Spectral_clustering(ADJ_Matrix, nclusters=3, affinity_v='precomputed')
    Net_Fast_Unfolding(ADJ_Matrix)

