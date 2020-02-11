from argparse import ArgumentParser
from pandas import read_csv
from pandas import DataFrame
from os import environ
from os.path import join
from os.path import exists
from os.path import abspath
from ast import literal_eval
from numpy import asarray
from numpy import log
from numpy import logical_not
from numpy.random import seed
from scipy.ndimage import distance_transform_edt as dist_transform
from utils.train import train_all
from utils.train import train_cross_val
from utils.logger import conditional_makedir
from utils.transforms import DistanceMapTransform
from utils.data_structures import load_data
from utils.data_structures import ConfigParser
from utils.data_structures import DataStorage
from utils.data_structures import MetricsStorage

def main(config):
    seed(seed=config.seed)

    #### LOAD DATASETS ####
    dataset             = read_csv(join(config.data_path, 'Dataset.csv'), index_col=0)
    dataset             = dataset.sort_index(axis=1)
    labels              = asarray(list(dataset)) # In case no data augmentation is applied
    description         = dataset.describe()

    # Define the window of validity of the ground truth masks
    if exists(join(config.data_path, 'Validity.csv')):
        validity        = read_csv(join(config.data_path, 'Validity.csv'), index_col=0, 
                                   converters={"on": literal_eval, "off": literal_eval}).T
    else:
        validity        = DataFrame(dict(zip(dataset.keys(), [[[0],[225000]]]*len(dataset.keys()))),
                                           index=['on', 'off'])

    # Zero-center data
    for key in description:
        dataset[key]    = (dataset[key] - description[key]['mean'])/description[key]['std']

    # Initialize data structures for results storage
    data                = DataStorage(dataset, validity)
    results             = DataStorage(dataset, validity)
    results_CV2         = DataStorage(dataset, validity)

    # Initialize data storage
    data.init_P(
        read_csv(join(config.data_path, 'Pwave.csv'), index_col=0, dtype='float'),
        load_data(join(config.data_path, 'Pon.csv')),
        load_data(join(config.data_path, 'Ppeak.csv')),
        load_data(join(config.data_path, 'Poff.csv'))
    )

    data.init_QRS(
        read_csv(join(config.data_path, 'QRSwave.csv'), index_col=0, dtype='float'),
        load_data(join(config.data_path, 'QRSon.csv')),
        load_data(join(config.data_path, 'QRSpeak.csv')),
        load_data(join(config.data_path, 'QRSoff.csv'))
    )

    data.init_T(
        read_csv(join(config.data_path, 'Twave.csv'), index_col=0, dtype='float'),
        load_data(join(config.data_path, 'Ton.csv')),
        load_data(join(config.data_path, 'Tpeak.csv')),
        load_data(join(config.data_path, 'Toff.csv'))
    )

    # Store metrics
    metrics     = MetricsStorage()
    metrics_CV2 = MetricsStorage()

    #### TRAIN/TEST DISTRIBUTION IMPORTING ####
    if config.splitting.lower() == 'cross_validation':
        train_keys = read_csv('./CommonDistribution/TrainSplit_Labels.csv',index_col=0)
        test_keys  = read_csv('./CommonDistribution/TestSplit_Labels.csv',index_col=0)
        KFolds     = [tuple([train_keys.values[i,:],test_keys.values[i,:]]) for i in range(train_keys.shape[0])]

        #### TRAIN THE MODEL ####
        train_cross_val(KFolds, config, data, metrics, metrics_CV2, results, results_CV2)
    elif config.splitting.lower() == 'all':
        IDs = [labels[i].split("_")[0] for i in range(len(labels))]
        IDs = asarray(list(set(IDs))) # Avoid duplicates

        #### TRAIN THE MODEL ####
        train_all(config, data, metrics, metrics_CV2, results, results_CV2, IDs)
    else:
        raise ValueError("Train type not understood")


if __name__ == '__main__':
    # Retrieve input data
    parser = ArgumentParser()
    parser.add_argument('--evaluate',       type=str, default='no', help='whether to train or evaluate')
    parser.add_argument('--backend',        type=str, default='keras', help='backend to use')
    parser.add_argument('--data_path',      type=str, default='/media/guille/DADES/DADES/PhysioNet/QTDB', help='data directory path, absolute')
    parser.add_argument('--data_set',       type=str, default='automatic', help='which dataset to use, in {automatic, manual0, manual1}')
    parser.add_argument('--stride',         type=int, default=None, help='override default stride')
    parser.add_argument('--data_aug',       type=str, default=None, help='override data augmentation')
    parser.add_argument('--n_epochs',       type=str, default=None, help='override number of epochs')
    parser.add_argument('--output_dir',     type=str, default=None, help='output directory to store weights and evaluations')
    parser.add_argument('--config_path',    type=str, default='./Configurations.csv' , help='file with all considered configurations')    
    parser.add_argument('--splitting',      type=str, default='cross_validation', help='"cross_validation" or "all"')
    parser.add_argument('--load_weights',   type=str, default='no', help='Load existing weights? Useful for fine-tuning')
    parser.add_argument('--config',         type=int, help='specific configuration to run')
    
    inputs              = parser.parse_args()

    ex_id               = inputs.config
    data_set            = inputs.data_set
    override_stride     = inputs.stride
    override_augment    = inputs.data_aug
    override_epochs     = inputs.n_epochs
    data_path           = inputs.data_path
    config_path         = inputs.config_path
    output_dir          = inputs.output_dir
    backend             = inputs.backend    
    splitting           = inputs.splitting
    load_weights        = inputs.load_weights.lower() in ['1', 'true', 'yes', 'y']
    evaluate            = inputs.evaluate.lower() in ['1', 'true', 'yes', 'y']


    # Store 
    Configurations      = read_csv(config_path,index_col=0)
    ExecutionParams     = Configurations.T[int(ex_id)]

    if (override_augment != None) and (override_augment not in ['', 'None']) and (override_augment in ['1', 'true', 'yes', 'y']):
        ExecutionParams['T_DataAug'] = True
    if (override_stride != None) and (override_stride not in ['', 'None']):
        ExecutionParams['T_Stride']  = int(override_stride)
    if (override_epochs != None) and (override_epochs not in ['', 'None']):
        ExecutionParams['T_Epochs']  = int(override_epochs)

    #### PARAMETERS ####    
    config = ConfigParser(ExecutionParams, ex_id, data_path, data_set, splitting, load_weights, backend, output_dir, evaluate)

    # print("CURRENT CONFIGURATION: ")
    # print(" ")
    # print(ExecutionParams)
    # print(" ")
    # print("Data path:      " + config.data_path)
    # print("Output path:    " + config.output_dir)
    # print("Training split: " + config.splitting)
    # print("Train model:    " + str(not config.evaluate))
    # print("Load weights:   " + str(config.load_weights))
    # print("Backend:        " + config.backend)
    # print(" ")

    print(config)

    # Call main function
    main(config)


