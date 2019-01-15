import json
import pandas as pd
from tqdm import tqdm
import os


class JaccardWeighted:
    """带权Jaccard算法"""
    """目前实现了两个算法接口：similarity_weighted和similarity_weighted_topn，使用示例如下：
    from JaccardWeighted import JaccardWeighted
    jaccard = JaccardWeighted('data/patient_data/', 'data/weight.json', '人口学信息', 'gb2312')
    
    print(jaccard.similarity_weighted('ZY130000306099', 'ZY220000306099'))  # 查找两个患者的相似度
    #  输出：0.8218
    
    print(jaccard.similarity_weighted_topn('ZY130000306099', 10))  # 查找某个患者最相似的Top10患者
    #  输出：{'ZY210000306099': 0.901, 'ZY240000306099': 0.901, 'ZY170000306099': 0.8687, 'ZY010000182436': 0.8667, 
    'ZY250000283248': 0.8598, 'ZY020000516586': 0.8438, 'ZY190000306099': 0.8351, 'ZY220000279662': 0.8333, 
    'ZY290000306099': 0.8318, 'ZY220000306099': 0.8218}
    
    """

    def __init__(self, data_dir_path, weight_file_path, patients_id_file_name='人口学信息', encoding='gb2312',
                 select_files=None):
        """ 
        初始化函数
        :param data_dir_path: 患者csv数据文件夹地址
        :param weight_file_path: 权值json文件地址
        :param patients_id_file_name: 所有患者住院ID信息所在的文件名称
        :param encoding: 文件统一编码格式，默认位gb2312
        :param select_files: 指定的数据文件名称列表，不指定则默认使用文件夹中的所有数据文件
        """
        self.data_dir_path = data_dir_path
        self.weight = json.load(open(weight_file_path, mode='r', encoding=encoding))
        self.patients_id_file_path = data_dir_path + patients_id_file_name + '.csv'
        self.encoding = encoding
        self.patients_datas = self.__get_datas(data_dir_path, select_files)

    def similarity_weighted(self, id1, id2):
        """
        使用两个患者住院ID计算带权Jaccard相似度
        :param id1: 患者1的住院ID
        :param id2: 患者2的住院ID
        :return: 两个患者的带权Jaccard相似度（精确到小数点后4位）
        """
        patient1 = self.__get_patient_property(id1, self.patients_datas)
        if not patient1:
            print('没有找到患者住院ID(%s)的记录！' % id1)
        patient2 = self.__get_patient_property(id2, self.patients_datas)
        if not patient2:
            print('没有找到患者住院ID(%s)的记录！' % id2)
        return self.__jaccard_weighted(patient1, patient2)

    def similarity_weighted_topn(self, p_id, n=10):
        """
        使用某患者的住院ID找出带权Jaccard相似度最高的Top-n患者
        :param p_id: 该患者住院ID
        :param n: 最相似患者数量，默认为10
        :return: 按相似度降序排序的最多n条记录的字典：{患者住院ID:相似度}
        """
        patients = pd.read_csv(open(self.patients_id_file_path, mode='r', encoding=self.encoding))
        patients_id = patients[patients.iloc[:, 0] != p_id].iloc[:, 0].values.tolist()
        patient1 = self.__get_patient_property(p_id, self.patients_datas)
        if not patient1:
            print('没有找到患者住院ID(%s)的记录！' % p_id)
            return {}
        p2_sim = {}
        for i in tqdm(range(len(patients_id)), desc='Finding...'):
            patient2 = self.__get_patient_property(patients_id[i], self.patients_datas)
            p2_sim[patients_id[i]] = self.__jaccard_weighted(patient1, patient2)
        return dict(sorted(p2_sim.items(), key=lambda x: x[1], reverse=True)[:n])

    def __jaccard_weighted(self, patient1, patient2):
        """
        计算带权Jaccard相似度
        :param patient1: 患者1信息字典
        :param patient2: 患者2信息字典
        :return: 两个患者的带权Jaccard相似度（精确到小数点后4位）
        """
        intersection = 0.0
        union = 0.0
        for key in list(patient1.keys()) + list(patient2.keys()):
            if key in self.weight.keys():
                union += self.weight[key]
                if key in patient1.keys() and key in patient2.keys() and patient1[key] == patient2[key]:
                    intersection += self.weight[key]
        return float('%.4f' % (intersection / union)) if union else union

    def __get_patient_property(self, p_id, patients_datas):
        """
        利用患者住院ID获取该患者的信息
        :param p_id: 患者住院ID
        :param patients_datas: 包含Dataframe格式数据的文件列表
        :return: 该患者的信息字典
        """
        patient_data = {}
        for datas in patients_datas:
            find = datas[datas.iloc[:, 0] == p_id]
            if len(find):
                data = find.iloc[0, 1:].to_dict()
                for key, value in data.items():
                    if value not in ['', ' ', '\n', '\t', '\r', '无']:
                        patient_data[key] = value
        return patient_data

    def __get_datas(self, dir_path, select_files=None):
        """
        获取文件夹中的所有数据，默认使用.csv格式的文件，用pandas读取
        :param dir_path: 数据文件夹地址
        :param select_files: 指定的数据文件名称列表，不指定则默认使用文件夹中的所有数据文件
        :return: 包含Dataframe格式数据的文件列表
        """
        patients_datas = []
        files = self.__walk_dir(dir_path)
        for file in files[2]:
            fname, ftype = os.path.splitext(file)
            if (ftype not in ['.csv', '.CSV']) or (select_files and fname not in select_files):
                continue
            data_file = open(os.path.join(files[0], file), mode='r', encoding=self.encoding)
            patients_datas.append(pd.read_csv(data_file).fillna(''))
        return patients_datas

    def __walk_dir(self, root):
        """
        获取文件夹中文件信息
        :param root: 文件夹地址
        :return: 路径，文件夹列表， 文件列表
        """
        for path, dirs, files in os.walk(root):
            return path, dirs, files


if __name__ == '__main__':
    jaccard = JaccardWeighted('data/patient_data/', 'data/weight.json', '人口学信息', 'gb2312')
    
    print(jaccard.similarity_weighted('ZY130000306099', 'ZY220000306099'))  # 查找两个患者的相似度

    # print(jaccard.similarity_weighted_topn('ZY130000306099', 10))  # 查找某个患者最相似的Top10患者
