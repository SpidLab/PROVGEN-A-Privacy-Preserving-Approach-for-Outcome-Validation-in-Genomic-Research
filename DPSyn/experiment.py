# pylint: disable = W0614, W0401, C0411
# the above errcodes correspond to unused wildcard import, wildcard import, wrong-import-order
# In fact, we can run pylint in cmd and set options like: pylint --disable=Cxxxx,Wxxxx yyyy.py zzzz.py
import argparse
import copy
import sys
from pathlib import Path
from loguru import logger
import numpy as np

from enum import Enum
class DT:
    def __init__(self, path, case, ctrl, ep_range, count_range):
        self.file_path = path
        self.case_ids = case
        self.control_ids = ctrl
        self.ep_range = ep_range
        self.count_range = count_range
        
class DATA(Enum):
    lactose = 'lactose'
    hair = 'hair'
    hand = 'hand'
    eye = 'eye'
    
def init_dataset():
    global dataset
    
    dataset = {}
    dataset[DATA.lactose] = DT('''raw/D1 - Lactose intolerance/mixture_chrm2_split_0.csv''', 
                             ['7322', '7006', '6991', '6936', '6541', '6526', '6304', '6203', '6178', '6157', '6098', '6045', '6038', '6002', '5962', '5915', '5907', '5729', '5584', '5552', '5501', '5462', '5377', '5353', '5303', '5202', '5144', '5113', '5058', '5002', '4994', '4990', '4987', '4857', '4826', '4796', '4789', '4686', '4671', '4643', '4581', '4386', '4337', '4291', '4242', '4022', '4018', '3914', '3910', '3777', '3760', '3756', '3749', '3722', '3689', '3460', '3349', '3227', '3203', '3132'],
                             ['7249', '7191', '7141', '7123', '7075', '7067', '7065', '7013', '6957', '6896', '6865', '6715', '6658', '6651', '6596', '6542', '6521', '6429', '6405', '6340', '6330', '6234', '5960', '5957', '5932', '5850', '5842', '5717', '5605', '5578', '5529', '5488', '5426', '5370', '5333', '5242', '5241', '5234', '5087', '5043', '4946', '4918', '4912', '4820', '4736', '4646', '4640', '4624', '4610', '4376', '4372', '4304', '4159', '4072', '3995', '3971', '3943', '3892', '3815', '3792'],
                             [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                              [0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
    dataset[DATA.hair] = DT('''raw/D2 - Hair color/mixture_split_0.csv''', 
                               ['1033', '1040', '1269', '1459', '1470', '1500', '1537', '1627', '175', '1764', '187', '1902', '1963', '1980', '2238', '2291', '2498', '2507', '2510', '252', '2557', '262', '266', '2660', '268', '2745', '2758', '285', '294', '296', '2964', '330', '366', '380', '397', '437', '4672', '481', '503', '533', '55', '567', '60', '64', '644', '646', '650', '678', '693', '707', '726', '754', '762', '775', '785', '792', '806', '807', '854', '920'],
                               ['1', '1028', '1039', '1051', '1085', '1103', '1121', '1424', '1494', '1718', '1964', '1968', '1970', '2158', '2454', '2681', '2724', '2759', '2797', '2961', '2969', '3046', '341', '3503', '3531', '3625', '3773', '3865', '3908', '3923', '4106', '45', '4695', '4962', '5284', '5425', '5489', '5515', '5741', '583', '6155', '63', '6356', '667', '721', '7359', '748', '749', '782', '7876', '8073', '8178', '8416', '842', '8598', '887', '915', '9323', '9324', '972'],
                               [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                             [0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
    dataset[DATA.hand] = DT('''raw/D3 - Handedness/mixture_chrm2.csv''',
                              ['11', '118', '123', '13', '139', '14', '146', '159', '16', '177', '187', '200', '202', '216', '22', '251', '252', '262', '266', '276', '279', '295', '328', '329', '33', '332', '341', '345', '347', '380', '40', '411', '441', '45', '468', '473', '478', '502', '503', '53', '574', '574', '583', '589', '6', '60', '602', '610', '613', '63', '645', '646', '650', '667', '672', '684', '693', '732', '75', '77'],
                              ['1010', '1060', '1066', '1112', '1123', '1325', '1525', '1764', '180', '1994', '2074', '2131', '2274', '2498', '2826', '2887', '294', '296', '2966', '3034', '3065', '3167', '323', '3566', '3622', '3936', '3965', '4019', '4036', '4140', '4257', '4346', '44', '4943', '4985', '528', '5310', '5475', '55', '5515', '594', '6331', '64', '704', '7360', '7403', '7700', '775', '7923', '8089', '81', '8202', '8414', '8701', '8883', '9112', '916', '9215', '9251', '9616'],
                              [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                            [0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
    dataset[DATA.eye] = DT('''raw/D4 - Eye color/mixture_chrm15.csv''',
                              ['10', '10012', '10074', '1010', '10142', '1026', '1033', '1034', '1036', '1071', '11', '1114', '1121', '1125', '1142', '1163', '1173', '118', '1209', '1221', '1233', '1235', '126', '1281', '1288', '13', '1375', '1382', '1384', '1392', '14', '141', '1420', '144', '146', '1466', '1467', '1470', '1532', '1537', '154', '1566', '1627', '1668', '1678', '175', '1759', '1785', '1825', '187', '1877', '1879', '1885', '189', '1952', '1963', '1975', '1987', '1991', '1996', '2008', '2018', '2035', '2075', '2077', '2133', '2153', '2162', '2217', '2236', '2239', '2276', '2279', '2337', '2371', '2385', '2402', '2421', '2455', '2512', '2513', '2535', '2557', '2576', '258', '2616', '262', '2628', '2645', '266', '2675', '268', '2702', '2725', '2759', '276', '279', '2841', '2849', '285', '2858', '2887', '2901', '2907', '2949', '2960', '2964', '2966', '298', '2998', '3046', '3069', '311', '3130', '3140', '3196', '322', '323', '3257', '328', '33', '3335', '3375', '3381', '3394', '3397', '3399', '3468', '3486', '3499', '3547', '357', '3581', '3605', '3615', '3622', '3625', '3639', '366', '3689', '3695', '3766', '3780', '380', '3835', '3853', '3883', '3904', '3905', '3913', '3921', '3922', '3931', '3934', '3964', '3965', '397', '3973', '3979', '3992', '4006', '4018', '4029', '4032', '4034', '4048', '4066', '4084', '4127', '4171', '4185', '4230', '4346', '4381', '44', '440', '4411', '4428', '4490', '4507', '4569', '4577', '4580', '4605', '4632', '4643', '4646', '466', '4671', '4672', '4680', '4715', '4758', '481', '4813', '4814', '483', '4833', '4837', '4851', '4859', '488', '4897', '4943', '4990', '4998', '5105', '5146', '5167', '5182', '5185', '5195', '5207', '5263', '5288', '5302', '5310', '5332', '539', '5408', '5442', '5460', '549', '5515', '554', '5557', '556', '5581', '561', '563', '5675', '5719', '5723', '5745', '5803', '5830', '5833', '5881', '589', '5924', '5973', '5989', '6020', '6064', '6065', '6074', '6099', '610', '6104', '6120', '6124', '613', '6192', '62', '6249', '6356', '644', '6484', '6573', '6583', '6589', '6721', '6733', '6734', '675', '678', '6815', '6858', '6859', '6869', '6873', '688', '6882', '6936', '6981', '6993', '7020', '7047', '7055', '7080', '7092', '7144', '7153', '7156', '7195', '722', '726', '7285', '7312', '7358', '7360', '7379', '7405', '7413', '7446', '745', '7471', '7475', '748', '7485', '75', '7504', '754', '7547', '7549', '7565', '7585', '7610', '7611', '762', '763', '7759', '777', '7783', '7905', '792', '7929', '7950', '7954', '7955', '798', '7989', '799', '8', '803', '8046', '807', '8089', '813', '8134', '8138', '8155', '816', '8178', '8184', '820', '8203', '8266', '8291', '8314', '832', '8336', '8387', '8415', '8417', '8423', '845', '8522', '854', '8563', '8598', '862', '8635', '8737', '8787', '8788', '8803', '8808', '8818', '8841', '8852', '8874', '8890', '8988', '9078', '9133', '916', '9166', '918', '9185', '920', '922', '9220', '924', '9246', '926', '9262', '9375', '9399', '943', '9450', '9457', '951', '9516', '9527', '954', '9565', '9635', '966', '9697', '9702', '9720', '9726', '9743', '9747', '9799', '9803', '9812', '9839', '9869', '9969'],
                              ['1', '10080', '10123', '1013', '1020', '1022', '1028', '1029', '1039', '1040', '1059', '1066', '1075', '1077', '1085', '1089', '1103', '1104', '1129', '1131', '1133', '1134', '1139', '1147', '1165', '1177', '1190', '1196', '123', '124', '1269', '1275', '1283', '1312', '1325', '1379', '1424', '1426', '1445', '1459', '1494', '1499', '1503', '1514', '1531', '1588', '159', '16', '160', '1600', '161', '168', '17', '1717', '1764', '177', '180', '1802', '1833', '1875', '1964', '1968', '1970', '1980', '200', '2000', '2004', '202', '2023', '2024', '2030', '204', '2056', '207', '2074', '2080', '2099', '210', '2116', '215', '2158', '216', '2166', '2177', '2183', '22', '2202', '2212', '2238', '2249', '2274', '2278', '2288', '2291', '2293', '2297', '2307', '2314', '2334', '2350', '2356', '2361', '2362', '2367', '241', '2412', '2453', '2454', '2456', '2457', '2498', '2507', '251', '252', '253', '2533', '2542', '2554', '2568', '2570', '26', '2633', '2648', '2660', '2663', '2681', '2687', '2700', '2707', '2718', '2720', '2739', '2764', '2797', '2826', '287', '2874', '288', '2881', '2886', '2890', '2920', '294', '295', '2958', '296', '2969', '2980', '2996', '3004', '3018', '3034', '305', '3078', '3090', '3104', '3167', '3186', '319', '325', '3265', '3275', '3280', '3289', '330', '3321', '3324', '3326', '337', '341', '3417', '3422', '3449', '345', '3454', '347', '3478', '349', '3494', '35', '352', '3531', '3532', '3543', '3569', '3582', '3598', '36', '3611', '363', '3642', '3643', '3644', '3655', '368', '3682', '3706', '3782', '38', '3821', '3834', '3847', '3865', '3881', '3882', '3908', '3930', '394', '3942', '3954', '3967', '40', '4009', '4030', '4036', '4038', '4055', '4056', '4057', '4080', '4095', '4106', '411', '4143', '4147', '4159', '4173', '4181', '4192', '42', '420', '4200', '429', '4349', '4356', '4359', '437', '439', '4399', '441', '4438', '4441', '4463', '4482', '4499', '4539', '4547', '4549', '4587', '4594', '4621', '463', '4648', '4679', '468', '4694', '4695', '4712', '473', '4756', '4797', '4801', '4825', '4879', '4932', '4935', '4941', '495', '4962', '497', '4983', '4985', '500', '502', '503', '5048', '5049', '5073', '5098', '5107', '5155', '5164', '5168', '5176', '5180', '5213', '5223', '5233', '5235', '5249', '5257', '5274', '5326', '533', '5342', '5397', '5425', '5432', '5447', '5456', '5475', '5478', '5487', '5495', '55', '5529', '5540', '5568', '5582', '5612', '5613', '5629', '567', '5683', '5701', '572', '574', '5741', '5743', '58', '5800', '581', '5820', '5821', '583', '5850', '5853', '5868', '5884', '5900', '5917', '595', '60', '6005', '6013', '602', '6035', '6045', '607', '609', '6103', '6115', '6187', '6191', '6206', '6238', '6239', '6244', '6248', '6269', '6284', '6285', '63', '6335', '6351', '636', '6363', '6367', '637', '6381', '64', '645', '646', '6483', '6491', '650', '651', '6545', '6560', '6617', '667', '669', '6690', '6699', '6706', '6731', '6758', '682', '684', '6851', '6861', '6866', '6889', '6910', '6928', '693', '6943', '6965', '6971', '6985', '6989', '7014', '704', '7053', '7070', '7072', '7082', '7089', '7098', '7114', '7155', '7163', '717', '7198', '7199', '721', '7214', '7218', '7232', '7252', '7266', '7280', '7292', '7320', '7322', '7393', '74', '7403', '7450', '7466', '749', '7510', '7520', '7568', '758', '761', '7636', '7639', '7657', '7685', '77', '7701', '775', '776', '7761', '7785', '7787', '779', '781', '782', '7841', '7876', '7915', '7923', '7945', '7967', '7981', '7992', '8058', '806', '8060', '8073', '8076', '808', '8090', '81', '8100', '811', '8129', '8135', '814', '8202', '8206', '8237', '8246', '8301', '8318', '836', '8363', '839', '8403', '8416', '842', '8449', '8463', '8476', '850', '8554', '8561', '865', '8675', '8701', '8731', '8796', '8864', '887', '8905', '8911', '8915', '8934', '8948', '8951', '8977', '898', '90', '903', '9072', '909', '9119', '912', '9126', '9135', '915', '9158', '9172', '9215', '9232', '925', '9276', '9323', '9324', '9351', '9367', '9372', '9380', '9421', '9427', '9430', '9486', '9489', '949', '9518', '9519', '952', '9541', '9561', '9568', '9589', '9616', '9672', '968', '972', '9749', '9781', '9796', '9871', '9876', '99', '990', '9920', '9928'],
                              [0.1, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
                            [0.1, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
init_dataset()

# print(sys.argv) 
# # assert False
# dataset_name = DATA.lactose
# ep=1.0
# snp_cnt = 100

# print(args)
# assert False
# PRIV_DATA = f'../datasets/{dataset_name.value}/{dataset_name.value}_{snp_cnt}.csv'
# PRIV_DATA_NAME = f'{dataset_name.value}_{snp_cnt}_{ep}'
# CONFIG_DATA = './config/data.yaml'
# PARAMS = f'./parameters/{dataset_name.value}/{ep}/parameters.json'
# DATA_TYPE = f'./parameters/{dataset_name.value}/{ep}/column_datatypes.json'
# MARGINAL_CONFIG = f'./config/eps={ep}.yaml'
# UPDATE_ITERATIONS = 30
# TARGET_PATH = f'./dpsyn_release/{dataset_name.value}/'
# TARGET_FILE_PATH = f'./dpsyn_release/{dataset_name.value}/{dataset_name.value}_{snp_cnt}_{ep}.csv'

parser = argparse.ArgumentParser()
# original dataset file 
parser.add_argument("--priv_data", type=str, default="./data/accidential_drug_deaths.csv",
                    help="specify the path of original data file in csv format")

# priv_data_name for use of naming mile-stone files
parser.add_argument("--priv_data_name", type=str, 
help="users must specify it to help mid-way naming and avoid possible mistakings")

# config file which include identifier and binning settings 
parser.add_argument("--config", type=str, default="./config/data.yaml",
                    help="specify the path of config file in yaml format")

# the default number of records is set as 100
parser.add_argument("--n", type=int, default=0, 
                    help="specify the number of records to generate")

# params file which include schema of the original dataset
parser.add_argument("--params", type=str, default="./data/parameters.json",
                    help="specify the path of parameters file in json format")

# datatype file which include the data types of the columns
parser.add_argument("--datatype", type=str, default="./data/column_datatypes.json",
                    help="specify the path of datatype file in json format")

# marginal_config which specify marginal usage method
parser.add_argument("--marginal_config", type=str, default="./config/eps=10.0.yaml",
help="specify the path of marginal config file in yaml format")

# hyper parameter, the num of update iterations
parser.add_argument("--update_iterations", type=int, default=30,
                   help="specify the num of update iterations")

# target path of synthetic dataset
parser.add_argument("--target_path", type=str, default="out.csv",
help="specify the target path of the synthetic dataset")

# target path of synthetic dataset
parser.add_argument("--synthetic_count", type=int, default=10,
help="specify the number of synthetic records to be generated")


args = parser.parse_args()
PRIV_DATA = args.priv_data
PRIV_DATA_NAME = args.priv_data_name
CONFIG_DATA = args.config
PARAMS = args.params
DATA_TYPE = args.datatype
MARGINAL_CONFIG = args.marginal_config
UPDATE_ITERATIONS = args.update_iterations
TARGET_FILE_PATH = args.target_path
TARGET_PATH = '/'.join(TARGET_FILE_PATH.split('/', -1)[:-1])
SYNTHETIC_COUNT = args.synthetic_count



from data.DataLoader import *
from data.RecordPostprocessor import RecordPostprocessor
from method.dpsyn import DPSyn


def main():
    np.random.seed(0)
    np.random.RandomState(0)
    # print(CONFIG_DATA)
    with open(CONFIG_DATA, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    # print(config)
    # assert False
    # dataloader initialization
    dataloader = DataLoader()
    dataloader.load_data()

    # default method is dpsyn
    method = 'dpsyn'

    n = SYNTHETIC_COUNT
    priv_data = PRIV_DATA
    priv_data_name = PRIV_DATA_NAME
   
    syn_data = run_method(config, dataloader, n)
    # if users set the records' num, we denote it in synthetic dataset's name
    if n!=0:
        print("------------------------> now we synthesize a dataset with ", n, "rows")
        Path(Path(TARGET_PATH)).mkdir(parents=True, exist_ok=True)
        syn_data.to_csv(Path(TARGET_FILE_PATH), index=False)
    # the default synthetic dataset name when n=0 
    else:
        syn_data.to_csv(Path(TARGET_FILE_PATH), index=False)


def run_method(config, dataloader, n):
    parameters = json.loads(Path(PARAMS).read_text())
    syn_data = None
    # each item in 'runs' specify one dp task with (eps, delta, sensitivity) 
    # as well as a possible 'max_records' value which bounds the dataset's size
    for r in parameters["runs"]:
        # 'max_records_per_individual' is the global sensitivity value of the designed function f
        #  here in the example f is the count, and you may change as you like
        eps, delta, sensitivity = r['epsilon'], r['delta'], r['max_records_per_individual']

        # we import logger in synthesizer.py
        # we import DPSyn which inherits synthesizer 
        logger.info(f'working on eps={eps}, delta={delta}, and sensitivity={sensitivity}')

        # we will use dpsyn to generate a dataset 
        """I guess it helps by displaying the runtime logic below
        1. DPSyn(Synthesizer)
        it got dataloader, eps, delta, sensitivity
        however, Synthesizer is so simple and crude(oh no it initializes the parameters in __init__)
        2. we call synthesizer.synthesize(fixed_n=n) which is written in dpsyn.py
        3. look at synthesize then
            def synthesize(self, fixed_n=0) -> pd.DataFrame:
            # def obtain_consistent_marginals(self, priv_marginal_config, priv_split_method) -> Marginals:
                noisy_marginals = self.obtain_consistent_marginals()
        4. it calls get_noisy_marginals() which is written in synthesizer.py
            # noisy_marginals = self.get_noisy_marginals(priv_marginal_config, priv_split_method)
        5. look at get_noisy_marginals()
            # we firstly generate punctual marginals
            priv_marginal_sets, epss = self.data.generate_marginal_by_config(self.data.private_data, priv_marginal_config)
            # todo: consider fine-tuned noise-adding methods for one-way and two-way respectively?
            # and now we add noises to get noisy marginals
            noisy_marginals = self.anonymize(priv_marginal_sets, epss, priv_split_method)
        6. look at generate_marginal_by_config() which is written in DataLoader.py
            we need config files like 
        e.g.3.
            priv_all_one_way: (or priv_all_two_way)
            total_eps: xxxxx
        7. look at anonymize() which is written in synthesizer.py 
            def anonymize(self, priv_marginal_sets: Dict, epss: Dict, priv_split_method: Dict) -> Marginals:
            noisy_marginals = {}
            for set_key, marginals in priv_marginal_sets.items():
                eps = epss[set_key]
            # noise_type, noise_param = advanced_composition.get_noise(eps, self.delta, self.sensitivity, len(marginals))
                noise_type = priv_split_method[set_key]
            (1)priv_split_method is hard_coded 
            (2) we decide the noise type by advanced_compisition()


        """
        synthesizer = DPSyn(dataloader, eps, delta, sensitivity)
        # tmp returns a DataFrame
        tmp = synthesizer.synthesize(fixed_n=n)
        
        # we add in the synthesized dataframe a new column which is 'epsilon'
        # so when do comparison, you should remove this column for consistence
        tmp['epsilon'] = eps

        # syn_data is a list, tmp is added in the list 
        if syn_data is None:
            syn_data = tmp
        else:
            syn_data = syn_data.append(tmp, ignore_index=True)


    # post-processing generated data, map records with grouped/binned attribute back to original attributes
    print("********************* START POSTPROCESSING ***********************")
    postprocessor = RecordPostprocessor()
    syn_data = postprocessor.post_process(syn_data, CONFIG_DATA, dataloader.decode_mapping)
    logger.info("------------------------>synthetic data post-processed:")
    print(syn_data)

    return syn_data


if __name__ == "__main__":   

    main()