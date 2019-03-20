import pandas as pd
import numpy as np
from keras.models import load_model
from keras.layers import *
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing


def loadEmbedding():
    embedding_file = open("./data/Embedding_100d.txt", 'r', encoding='UTF8')
    embedding = dict()
    for i in embedding_file:
        embedding[i.split()[0]] = np.asanyarray(i.split()[1:])
    return embedding


def sen2vec(i, embedding, max_len=30):
    maxLen = max_len
    ret = np.zeros((maxLen, 100))
    l = list(i)
    for j in range(0, min(len(l), maxLen)):
        if l[j] in embedding.keys():
            tem = embedding[l[j]]
            ret[j] = tem
    return ret


def preprocess(file):
    xls_file = pd.ExcelFile(file)
    data = xls_file.parse(0)[["患者住院ID", "年龄", "性别", "1年内急性加重次数", "吸烟史", "近期咳嗽", "近期咳痰",
                              "近期胸闷", "近期喘息", "近期气促", "既往气促", "夜间阵发性呼吸困难", "双下肢水肿",
                              "心悸", "血嗜酸性粒细胞计数", "主诉", "现病史"]][:]

    embedding = loadEmbedding()
    zhusu = load_model("./data/autoencoder_主诉.h5")
    layer_output_zhusu = K.function([zhusu.layers[0].input], [zhusu.layers[2].output])

    xianbingshi_class = load_model("./data/classfier_现病史.h5")
    layer_output_xianbingshi = K.function([xianbingshi_class.layers[0].input], [xianbingshi_class.layers[1].output])

    id = []
    structured_part = []
    unstructured_part = []
    for index, row in data.iterrows():
        #print(index)
        id.append(row["患者住院ID"])
        ###
        tem = []
        #tem.append(int(int(row["年龄"].split("岁")[0]) / 5) - 10)
        tem.append(0 if data.loc[index, "性别"] == "男" else 1)
        tem.append(0 if data.loc[index, "1年内急性加重次数"] != "过去1年急性加重>=2次或有1次住院史" else 1)
        tem.append(0 if str(data.loc[index, "吸烟史"]).split("吸烟史")[0] != "有" else 1)
        tem.append(0 if data.loc[index, "近期咳嗽"] != "是" else 1)
        tem.append(0 if data.loc[index, "近期咳痰"] != "是" else 1)
        tem.append(0 if data.loc[index, "近期胸闷"] != "是" else 1)
        tem.append(0 if data.loc[index, "近期喘息"] != "是" else 1)
        tem.append(0 if data.loc[index, "近期气促"] != "是" else 1)
        tem.append(0 if data.loc[index, "既往气促"] != "是" else 1)
        tem.append(0 if data.loc[index, "夜间阵发性呼吸困难"] != "是" else 1)
        tem.append(0 if data.loc[index, "双下肢水肿"] != "是" else 1)
        tem.append(0 if data.loc[index, "心悸"] != "是" else 1)
        if data.loc[index, "血嗜酸性粒细胞计数"] == ">=0.45":
            tem.append(2)
        elif data.loc[index, "血嗜酸性粒细胞计数"] == ">=0.35~0.44":
            tem.append(1)
        else:
            tem.append(0)
        tem = np.asarray(tem)
        structured_part.append(tem)
        ###
        line = []
        sentence = str(row["主诉"]).strip()
        tem = np.zeros(233)
        for sub in re.split(r"。|；|;|，|,", sentence):
            if sub == "":
                continue
            vec = sen2vec(sub, embedding, 30).reshape(1, -1, 100)
            tem = tem + layer_output_zhusu([vec])[0]
        line.append(tem[0])
        ###
        sentence = str(row["现病史"]).strip()
        predict_set = []
        for sub in re.split(r"。|；|;|，|,", sentence):
            if sub == "":
                continue
            vec = sen2vec(sub, embedding, 100)
            predict_set.append(vec)
        predict_set = np.asarray(predict_set)
        pred = xianbingshi_class.predict(predict_set).argmax(axis=-1)
        layer_output = layer_output_xianbingshi([predict_set])[0]
        representation = [np.zeros(128)] * (4)
        for i in range(0, len(pred)):
            representation[int(pred[i])] = representation[int(pred[i])] + layer_output[i]
        '''
        for i in range(0, max(pred) + 1):
            if np.sum(pred == i) != 0:
                representation[i] = representation[i] / (np.sum(pred == i))
        '''
        for i in range(0, len(representation)):
            line.append(representation[i])
        unstructured_part.append(line)
    d = dict()
    d["id"] = id
    d["structured_part"] = structured_part
    d["unstructured_part"] = unstructured_part
    return d


def compute_sim(file):
    data = preprocess(file)
    sim_s = cosine_similarity(data["structured_part"])
    sim_s = preprocessing.scale(sim_s)
    tem = []
    for i in data["unstructured_part"]:
        line = []
        for j in i:
            line.extend(j.tolist())
        tem.append(line)
    sim_u = cosine_similarity(tem)
    sim_u = preprocessing.scale(sim_u)
    sim = 0.5 * sim_s + 0.5 * sim_u
    return [data["id"], -sim[:]]


if __name__ == '__main__':
    # 只放了50个病人的数据，3000个病人运行要几分钟
    file = "./data/患者信息整合表.xls"
    # id是N个病人住院id序列 sim是N个病人的相似度矩阵 (N*N)
    [id, sim] = compute_sim(file)
    print(id, sim)
