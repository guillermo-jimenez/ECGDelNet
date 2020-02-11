import numpy as np
import wfdb
import os
import os.path
import pandas
import glob
from utils.data_structures import save_fiducials

dataDir = '/media/guille/DADES/DADES/PhysioNet/QTDB/'
# dataDir = '/homedtic/gjimenez/DADES/PhysioNet/QTDB/'

dataset = pandas.DataFrame()
records           = glob.glob(dataDir + '*.dat')
lengths           = np.zeros(len(records),)
nsig              = np.zeros(len(records),)
maxSize           = 225000

# Retrieve the signals    
for filename in records:
    fname = os.path.splitext(os.path.split(filename)[1])[0]

    # Read record
    rec = wfdb.rdrecord(dataDir + fname)
    dataset[fname + '_0']               = (rec.adc()[-1,0] - rec.baseline[0])/np.asarray(rec.adc_gain[0])*np.ones((maxSize,))
    dataset[fname + '_1']               = (rec.adc()[-1,1] - rec.baseline[1])/np.asarray(rec.adc_gain[1])*np.ones((maxSize,))

    dataset[fname + '_0'][:rec.sig_len] = (rec.adc()[:,0] - rec.baseline[0])/np.asarray(rec.adc_gain[0])
    dataset[fname + '_1'][:rec.sig_len] = (rec.adc()[:,1] - rec.baseline[1])/np.asarray(rec.adc_gain[1])

dataset.to_csv(os.path.join(dataDir,'ToCSV','Dataset.csv'))

QRSon_automatic   = dict()
QRSpeak_automatic = dict()
QRSoff_automatic  = dict()

Pon_automatic     = dict()
Ppeak_automatic   = dict()
Poff_automatic    = dict()

Ton_automatic     = dict()
Tpeak_automatic   = dict()
Toff_automatic    = dict()

QRSon_manual0     = dict()
QRSpeak_manual0   = dict()
QRSoff_manual0    = dict()

Pon_manual0       = dict()
Ppeak_manual0     = dict()
Poff_manual0      = dict()

Ton_manual0       = dict()
Tpeak_manual0     = dict()
Toff_manual0      = dict()

QRSon_manual1     = dict()
QRSpeak_manual1   = dict()
QRSoff_manual1    = dict()

Pon_manual1       = dict()
Ppeak_manual1     = dict()
Poff_manual1      = dict()

Ton_manual1       = dict()
Tpeak_manual1     = dict()
Toff_manual1      = dict()

Pwave_automatic   = pandas.DataFrame()
QRSwave_automatic = pandas.DataFrame()
Twave_automatic   = pandas.DataFrame()

Pwave_manual0     = pandas.DataFrame()
QRSwave_manual0   = pandas.DataFrame()
Twave_manual0     = pandas.DataFrame()

Pwave_manual1     = pandas.DataFrame()
QRSwave_manual1   = pandas.DataFrame()
Twave_manual1     = pandas.DataFrame()



for filename in records:
    fname = os.path.splitext(os.path.split(filename)[1])[0]

    ############# MANUAL ANNOTATIONS #############
    # First annotator & filter _out_ repeated T fiducials (some registries mark T and T prime)
    manual0 = wfdb.rdann(dataDir + fname, 'q1c', return_label_elements=['label_store', 'symbol'], summarize_labels=True)
    indexes = np.where(np.diff(manual0.label_store) == 0)[0]+1
    manual0.num = np.delete(manual0.num, indexes)
    manual0.label_store = np.delete(manual0.label_store, indexes)
    manual0.sample = np.delete(manual0.sample, indexes)
    
    # Second annotator & filter _out_ repeated T fiducials (some registries mark T and T prime)
    if os.path.exists(dataDir + fname + '.q2c'):
        manual1 = wfdb.rdann(dataDir + fname, 'q2c', return_label_elements=['label_store', 'symbol'], summarize_labels=True)
        indexes = np.where(np.diff(manual1.label_store) == 0)[0]+1
        manual1.num = np.delete(manual1.num, indexes)
        manual1.label_store = np.delete(manual1.label_store, indexes)
        manual1.sample = np.delete(manual1.sample, indexes)
    
    ############# AUTOMATIC ANNOTATIONS #############
    # First algorithm & filter _out_ repeated T fiducials (some registries mark T and T prime)
    automatic0 = wfdb.rdann(dataDir + fname, 'pu0', return_label_elements=['label_store', 'symbol'], summarize_labels=True)
    indexes = np.where(np.diff(automatic0.label_store) == 0)[0]+1
    automatic0.num = np.delete(automatic0.num, indexes)
    automatic0.label_store = np.delete(automatic0.label_store, indexes)
    automatic0.sample = np.delete(automatic0.sample, indexes)
    
    # Second algorithm & filter _out_ repeated T fiducials (some registries mark T and T prime)
    automatic1 = wfdb.rdann(dataDir + fname, 'pu1', return_label_elements=['label_store', 'symbol'], summarize_labels=True)
    indexes = np.where(np.diff(automatic1.label_store) == 0)[0]+1
    automatic1.num = np.delete(automatic1.num, indexes)
    automatic1.label_store = np.delete(automatic1.label_store, indexes)
    automatic1.sample = np.delete(automatic1.sample, indexes)
    
    ############# RETRIEVE BINARY MASKS #############
    maskP_automatic0   = np.zeros((maxSize,),dtype='int8')
    maskQRS_automatic0 = np.zeros((maxSize,),dtype='int8')
    maskT_automatic0   = np.zeros((maxSize,),dtype='int8')

    maskP_automatic1   = np.zeros((maxSize,),dtype='int8')
    maskQRS_automatic1 = np.zeros((maxSize,),dtype='int8')
    maskT_automatic1   = np.zeros((maxSize,),dtype='int8')

    maskP_manual0      = np.zeros((maxSize,),dtype='int8')
    maskQRS_manual0    = np.zeros((maxSize,),dtype='int8')
    maskT_manual0      = np.zeros((maxSize,),dtype='int8')

    maskP_manual1      = np.zeros((maxSize,),dtype='int8')
    maskQRS_manual1    = np.zeros((maxSize,),dtype='int8')
    maskT_manual1      = np.zeros((maxSize,),dtype='int8')

    # Automatic 0
    for i in range(automatic0.sample[np.where(automatic0.label_store == 24)[0]].size):
        maskP_automatic0[(automatic0.sample[np.where(automatic0.label_store == 24)[0]-1])[i]:(automatic0.sample[np.where(automatic0.label_store == 24)[0]+1])[i]] = True

    for i in range(automatic0.sample[np.where(automatic0.label_store == 1)[0]].size):
        maskQRS_automatic0[(automatic0.sample[np.where(automatic0.label_store == 1)[0]-1])[i]:(automatic0.sample[np.where(automatic0.label_store == 1)[0]+1])[i]] = True

    for i in range(automatic0.sample[np.where(automatic0.label_store == 27)[0]].size):
        maskT_automatic0[(automatic0.sample[np.where(automatic0.label_store == 27)[0]-1])[i]:(automatic0.sample[np.where(automatic0.label_store == 27)[0]+1])[i]] = True

    # Automatic 1
    for i in range(automatic1.sample[np.where(automatic1.label_store == 24)[0]].size):
        maskP_automatic1[(automatic1.sample[np.where(automatic1.label_store == 24)[0]-1])[i]:(automatic1.sample[np.where(automatic1.label_store == 24)[0]+1])[i]] = True

    for i in range(automatic1.sample[np.where(automatic1.label_store == 1)[0]].size):
        maskQRS_automatic1[(automatic1.sample[np.where(automatic1.label_store == 1)[0]-1])[i]:(automatic1.sample[np.where(automatic1.label_store == 1)[0]+1])[i]] = True

    for i in range(automatic1.sample[np.where(automatic1.label_store == 27)[0]].size):
        maskT_automatic1[(automatic1.sample[np.where(automatic1.label_store == 27)[0]-1])[i]:(automatic1.sample[np.where(automatic1.label_store == 27)[0]+1])[i]] = True


    # Manual 0
    for i in range(manual0.sample[np.where(manual0.label_store == 24)[0]].size):
        maskP_manual0[(manual0.sample[np.where(manual0.label_store == 24)[0]-1])[i]:(manual0.sample[np.where(manual0.label_store == 24)[0]+1])[i]] = True

    for i in range(manual0.sample[np.where(manual0.label_store == 1)[0]].size):
        maskQRS_manual0[(manual0.sample[np.where(manual0.label_store == 1)[0]-1])[i]:(manual0.sample[np.where(manual0.label_store == 1)[0]+1])[i]] = True

    for i in range(manual0.sample[np.where(manual0.label_store == 27)[0]].size):
        maskT_manual0[(manual0.sample[np.where(manual0.label_store == 27)[0]-1])[i]:(manual0.sample[np.where(manual0.label_store == 27)[0]+1])[i]] = True

    # Manual 1
    if os.path.exists(dataDir + fname + '.q2c'):
        for i in range(manual1.sample[np.where(manual1.label_store == 24)[0]].size):
            maskP_manual1[(manual1.sample[np.where(manual1.label_store == 24)[0]-1])[i]:(manual1.sample[np.where(manual1.label_store == 24)[0]+1])[i]] = True

        for i in range(manual1.sample[np.where(manual1.label_store == 1)[0]].size):
            maskQRS_manual1[(manual1.sample[np.where(manual1.label_store == 1)[0]-1])[i]:(manual1.sample[np.where(manual1.label_store == 1)[0]+1])[i]] = True

        for i in range(manual1.sample[np.where(manual1.label_store == 27)[0]].size):
            maskT_manual1[(manual1.sample[np.where(manual1.label_store == 27)[0]-1])[i]:(manual1.sample[np.where(manual1.label_store == 27)[0]+1])[i]] = True

    
    ############# CHOOSE ALGORITHM WITH HIGHEST DICE COEFFICIENT FOR THAT LEAD #############
    dice_P_00 = np.sum(maskP_manual0 * maskP_automatic0)*2.0 / (np.sum(maskP_manual0) + np.sum(maskP_automatic0))
    dice_P_01 = np.sum(maskP_manual0 * maskP_automatic1)*2.0 / (np.sum(maskP_manual0) + np.sum(maskP_automatic1))
    dice_P_10 = np.sum(maskP_manual1 * maskP_automatic0)*2.0 / (np.sum(maskP_manual1) + np.sum(maskP_automatic0))
    dice_P_11 = np.sum(maskP_manual1 * maskP_automatic1)*2.0 / (np.sum(maskP_manual1) + np.sum(maskP_automatic1))

    dice_QRS_00 = np.sum(maskQRS_manual0 * maskQRS_automatic0)*2.0 / (np.sum(maskQRS_manual0) + np.sum(maskQRS_automatic0))
    dice_QRS_01 = np.sum(maskQRS_manual0 * maskQRS_automatic1)*2.0 / (np.sum(maskQRS_manual0) + np.sum(maskQRS_automatic1))
    dice_QRS_10 = np.sum(maskQRS_manual1 * maskQRS_automatic0)*2.0 / (np.sum(maskQRS_manual1) + np.sum(maskQRS_automatic0))
    dice_QRS_11 = np.sum(maskQRS_manual1 * maskQRS_automatic1)*2.0 / (np.sum(maskQRS_manual1) + np.sum(maskQRS_automatic1))

    dice_T_00 = np.sum(maskT_manual0 * maskT_automatic0)*2.0 / (np.sum(maskT_manual0) + np.sum(maskT_automatic0))
    dice_T_01 = np.sum(maskT_manual0 * maskT_automatic1)*2.0 / (np.sum(maskT_manual0) + np.sum(maskT_automatic1))
    dice_T_10 = np.sum(maskT_manual1 * maskT_automatic0)*2.0 / (np.sum(maskT_manual1) + np.sum(maskT_automatic0))
    dice_T_11 = np.sum(maskT_manual1 * maskT_automatic1)*2.0 / (np.sum(maskT_manual1) + np.sum(maskT_automatic1))

    dice_0_P   = (dice_P_00 + dice_P_10)/2
    dice_0_QRS = (dice_QRS_00 + dice_QRS_10)/2
    dice_0_T   = (dice_T_00 + dice_T_10)/2

    dice_1_P   = (dice_P_01 + dice_P_11)/2
    dice_1_QRS = (dice_QRS_01 + dice_QRS_11)/2
    dice_1_T   = (dice_T_01 + dice_T_11)/2

    dice_0 = dice_0_P+dice_0_QRS+dice_0_T
    dice_1 = dice_1_P+dice_1_QRS+dice_1_T

    ############# STORE RESULTS FOR AUTOMATIC SEGMENTATION #############
    # Decide between the two automatic annotations for each wave
    # P wave
    if dice_0_P > dice_1_P:
        # STORE MASKS
        annotation = automatic0

        Pwave_automatic[fname + '_0']   = maskP_automatic0
        Pwave_automatic[fname + '_1']   = maskP_automatic0
    elif dice_0_P < dice_1_P:
        # STORE MASKS
        annotation = automatic1

        Pwave_automatic[fname + '_0']   = maskP_automatic1
        Pwave_automatic[fname + '_1']   = maskP_automatic1
    else:
        if dice_0 >= dice_1:
            # STORE MASKS
            annotation = automatic0

            Pwave_automatic[fname + '_0']   = maskP_automatic0
            Pwave_automatic[fname + '_1']   = maskP_automatic0
        else:
            # STORE MASKS
            annotation = automatic1

            Pwave_automatic[fname + '_0']   = maskP_automatic1
            Pwave_automatic[fname + '_1']   = maskP_automatic1

    # P fiducials
    Pon_automatic[fname + '_0']     = annotation.sample[np.where(annotation.label_store == 24)[0]-1]
    Pon_automatic[fname + '_1']     = annotation.sample[np.where(annotation.label_store == 24)[0]-1]
    Ppeak_automatic[fname + '_0']   = annotation.sample[np.where(annotation.label_store == 24)[0]]
    Ppeak_automatic[fname + '_1']   = annotation.sample[np.where(annotation.label_store == 24)[0]]
    Poff_automatic[fname + '_0']    = annotation.sample[np.where(annotation.label_store == 24)[0]+1]
    Poff_automatic[fname + '_1']    = annotation.sample[np.where(annotation.label_store == 24)[0]+1]

    # QRS wave
    if dice_0_QRS > dice_1_QRS:
        # STORE MASKS
        annotation = automatic0

        QRSwave_automatic[fname + '_0']   = maskQRS_automatic0
        QRSwave_automatic[fname + '_1']   = maskQRS_automatic0
    elif dice_0_QRS < dice_1_QRS:
        # STORE MASKS
        annotation = automatic1

        QRSwave_automatic[fname + '_0']   = maskQRS_automatic1
        QRSwave_automatic[fname + '_1']   = maskQRS_automatic1
    else:
        if dice_0 >= dice_1:
            # STORE MASKS
            annotation = automatic0

            QRSwave_automatic[fname + '_0']   = maskQRS_automatic0
            QRSwave_automatic[fname + '_1']   = maskQRS_automatic0
        else:
            # STORE MASKS
            annotation = automatic1

            QRSwave_automatic[fname + '_0']   = maskQRS_automatic1
            QRSwave_automatic[fname + '_1']   = maskQRS_automatic1

    # QRS fiducials
    QRSon_automatic[fname + '_0']     = annotation.sample[np.where(annotation.label_store == 1)[0]-1]
    QRSon_automatic[fname + '_1']     = annotation.sample[np.where(annotation.label_store == 1)[0]-1]
    QRSpeak_automatic[fname + '_0']   = annotation.sample[np.where(annotation.label_store == 1)[0]]
    QRSpeak_automatic[fname + '_1']   = annotation.sample[np.where(annotation.label_store == 1)[0]]
    QRSoff_automatic[fname + '_0']    = annotation.sample[np.where(annotation.label_store == 1)[0]+1]
    QRSoff_automatic[fname + '_1']    = annotation.sample[np.where(annotation.label_store == 1)[0]+1]

    # T wave
    if dice_0_T > dice_1_T:
        # STORE MASKS
        annotation = automatic0

        Twave_automatic[fname + '_0']   = maskT_automatic0
        Twave_automatic[fname + '_1']   = maskT_automatic0
    elif dice_0_T < dice_1_T:
        # STORE MASKS
        annotation = automatic1

        Twave_automatic[fname + '_0']   = maskT_automatic1
        Twave_automatic[fname + '_1']   = maskT_automatic1
    else:
        if dice_0 >= dice_1:
            # STORE MASKS
            annotation = automatic0

            Twave_automatic[fname + '_0']   = maskT_automatic0
            Twave_automatic[fname + '_1']   = maskT_automatic0
        else:
            # STORE MASKS
            annotation = automatic1

            Twave_automatic[fname + '_0']   = maskT_automatic1
            Twave_automatic[fname + '_1']   = maskT_automatic1

    # T fiducials
    Ton_automatic[fname + '_0']     = annotation.sample[np.where(annotation.label_store == 27)[0]-1]
    Ton_automatic[fname + '_1']     = annotation.sample[np.where(annotation.label_store == 27)[0]-1]
    Tpeak_automatic[fname + '_0']   = annotation.sample[np.where(annotation.label_store == 27)[0]]
    Tpeak_automatic[fname + '_1']   = annotation.sample[np.where(annotation.label_store == 27)[0]]
    Toff_automatic[fname + '_0']    = annotation.sample[np.where(annotation.label_store == 27)[0]+1]
    Toff_automatic[fname + '_1']    = annotation.sample[np.where(annotation.label_store == 27)[0]+1]

    ############# STORE RESULTS FOR MANUAL SEGMENTATION #############
    Pwave_manual0[fname + '_0'] = maskP_manual0
    Pwave_manual0[fname + '_1'] = maskP_manual0
    QRSwave_manual0[fname + '_0'] = maskQRS_manual0
    QRSwave_manual0[fname + '_1'] = maskQRS_manual0
    Twave_manual0[fname + '_0'] = maskT_manual0
    Twave_manual0[fname + '_1'] = maskT_manual0
    
    Pon_manual0[fname + '_0']     = manual0.sample[np.where(manual0.label_store == 24)[0]-1]
    Pon_manual0[fname + '_1']     = manual0.sample[np.where(manual0.label_store == 24)[0]-1]
    Ppeak_manual0[fname + '_0']   = manual0.sample[np.where(manual0.label_store == 24)[0]]
    Ppeak_manual0[fname + '_1']   = manual0.sample[np.where(manual0.label_store == 24)[0]]
    Poff_manual0[fname + '_0']    = manual0.sample[np.where(manual0.label_store == 24)[0]+1]
    Poff_manual0[fname + '_1']    = manual0.sample[np.where(manual0.label_store == 24)[0]+1]

    # QRS wave
    QRSon_manual0[fname + '_0']   = manual0.sample[np.where(manual0.label_store == 1)[0]-1]
    QRSon_manual0[fname + '_1']   = manual0.sample[np.where(manual0.label_store == 1)[0]-1]
    QRSpeak_manual0[fname + '_0'] = manual0.sample[np.where(manual0.label_store == 1)[0]]
    QRSpeak_manual0[fname + '_1'] = manual0.sample[np.where(manual0.label_store == 1)[0]]
    QRSoff_manual0[fname + '_0']  = manual0.sample[np.where(manual0.label_store == 1)[0]+1]
    QRSoff_manual0[fname + '_1']  = manual0.sample[np.where(manual0.label_store == 1)[0]+1]

    # T wave
    Ton_manual0[fname + '_0']     = manual0.sample[np.where(manual0.label_store == 27)[0]-1]
    Ton_manual0[fname + '_1']     = manual0.sample[np.where(manual0.label_store == 27)[0]-1]
    Tpeak_manual0[fname + '_0']   = manual0.sample[np.where(manual0.label_store == 27)[0]]
    Tpeak_manual0[fname + '_1']   = manual0.sample[np.where(manual0.label_store == 27)[0]]
    Toff_manual0[fname + '_0']    = manual0.sample[np.where(manual0.label_store == 27)[0]+1]
    Toff_manual0[fname + '_1']    = manual0.sample[np.where(manual0.label_store == 27)[0]+1]


    Pwave_manual1[fname + '_0'] = maskP_manual1
    Pwave_manual1[fname + '_1'] = maskP_manual1
    QRSwave_manual1[fname + '_0'] = maskQRS_manual1
    QRSwave_manual1[fname + '_1'] = maskQRS_manual1
    Twave_manual1[fname + '_0'] = maskT_manual1
    Twave_manual1[fname + '_1'] = maskT_manual1

    # Manual 1
    if os.path.exists(dataDir + fname + '.q2c'):

        Pon_manual1[fname + '_0']     = manual1.sample[np.where(manual1.label_store == 24)[0]-1]
        Pon_manual1[fname + '_1']     = manual1.sample[np.where(manual1.label_store == 24)[0]-1]
        Ppeak_manual1[fname + '_0']   = manual1.sample[np.where(manual1.label_store == 24)[0]]
        Ppeak_manual1[fname + '_1']   = manual1.sample[np.where(manual1.label_store == 24)[0]]
        Poff_manual1[fname + '_0']    = manual1.sample[np.where(manual1.label_store == 24)[0]+1]
        Poff_manual1[fname + '_1']    = manual1.sample[np.where(manual1.label_store == 24)[0]+1]

        # QRS wave
        QRSon_manual1[fname + '_0']   = manual1.sample[np.where(manual1.label_store == 1)[0]-1]
        QRSon_manual1[fname + '_1']   = manual1.sample[np.where(manual1.label_store == 1)[0]-1]
        QRSpeak_manual1[fname + '_0'] = manual1.sample[np.where(manual1.label_store == 1)[0]]
        QRSpeak_manual1[fname + '_1'] = manual1.sample[np.where(manual1.label_store == 1)[0]]
        QRSoff_manual1[fname + '_0']  = manual1.sample[np.where(manual1.label_store == 1)[0]+1]
        QRSoff_manual1[fname + '_1']  = manual1.sample[np.where(manual1.label_store == 1)[0]+1]

        # T wave
        Ton_manual1[fname + '_0']     = manual1.sample[np.where(manual1.label_store == 27)[0]-1]
        Ton_manual1[fname + '_1']     = manual1.sample[np.where(manual1.label_store == 27)[0]-1]
        Tpeak_manual1[fname + '_0']   = manual1.sample[np.where(manual1.label_store == 27)[0]]
        Tpeak_manual1[fname + '_1']   = manual1.sample[np.where(manual1.label_store == 27)[0]]
        Toff_manual1[fname + '_0']    = manual1.sample[np.where(manual1.label_store == 27)[0]+1]
        Toff_manual1[fname + '_1']    = manual1.sample[np.where(manual1.label_store == 27)[0]+1]
    else:
        Pon_manual1[fname + '_0']     = np.asarray([])
        Pon_manual1[fname + '_1']     = np.asarray([])
        Ppeak_manual1[fname + '_0']   = np.asarray([])
        Ppeak_manual1[fname + '_1']   = np.asarray([])
        Poff_manual1[fname + '_0']    = np.asarray([])
        Poff_manual1[fname + '_1']    = np.asarray([])

        # QRS wave
        QRSon_manual1[fname + '_0']   = np.asarray([])
        QRSon_manual1[fname + '_1']   = np.asarray([])
        QRSpeak_manual1[fname + '_0'] = np.asarray([])
        QRSpeak_manual1[fname + '_1'] = np.asarray([])
        QRSoff_manual1[fname + '_0']  = np.asarray([])
        QRSoff_manual1[fname + '_1']  = np.asarray([])

        # T wave
        Ton_manual1[fname + '_0']     = np.asarray([])
        Ton_manual1[fname + '_1']     = np.asarray([])
        Tpeak_manual1[fname + '_0']   = np.asarray([])
        Tpeak_manual1[fname + '_1']   = np.asarray([])
        Toff_manual1[fname + '_0']    = np.asarray([])
        Toff_manual1[fname + '_1']    = np.asarray([])
        

Pwave_automatic.to_csv(os.path.join(dataDir,'ToCSV','Pwave.csv'))
QRSwave_automatic.to_csv(os.path.join(dataDir,'ToCSV','QRSwave.csv'))
Twave_automatic.to_csv(os.path.join(dataDir,'ToCSV','Twave.csv'))

Pwave_manual0.to_csv(os.path.join(dataDir,'ToCSV','Pwave_q1c.csv'))
QRSwave_manual0.to_csv(os.path.join(dataDir,'ToCSV','QRSwave_q1c.csv'))
Twave_manual0.to_csv(os.path.join(dataDir,'ToCSV','Twave_q1c.csv'))

Pwave_manual1.to_csv(os.path.join(dataDir,'ToCSV','Pwave_q2c.csv'))
QRSwave_manual1.to_csv(os.path.join(dataDir,'ToCSV','QRSwave_q2c.csv'))
Twave_manual1.to_csv(os.path.join(dataDir,'ToCSV','Twave_q2c.csv'))

#### SAVE DATABASE ####
save_fiducials(Pon_automatic, os.path.join(dataDir,'ToCSV','Pon.csv'))
save_fiducials(Ppeak_automatic, os.path.join(dataDir,'ToCSV','Ppeak.csv'))
save_fiducials(Poff_automatic, os.path.join(dataDir,'ToCSV','Poff.csv'))
save_fiducials(QRSon_automatic, os.path.join(dataDir,'ToCSV','QRSon.csv'))
save_fiducials(QRSpeak_automatic, os.path.join(dataDir,'ToCSV','QRSpeak.csv'))
save_fiducials(QRSoff_automatic, os.path.join(dataDir,'ToCSV','QRSoff.csv'))
save_fiducials(Ton_automatic, os.path.join(dataDir,'ToCSV','Ton.csv'))
save_fiducials(Tpeak_automatic, os.path.join(dataDir,'ToCSV','Tpeak.csv'))
save_fiducials(Toff_automatic, os.path.join(dataDir,'ToCSV','Toff.csv'))

save_fiducials(Pon_manual0, os.path.join(dataDir,'ToCSV','Pon_q1c.csv'))
save_fiducials(Ppeak_manual0, os.path.join(dataDir,'ToCSV','Ppeak_q1c.csv'))
save_fiducials(Poff_manual0, os.path.join(dataDir,'ToCSV','Poff_q1c.csv'))
save_fiducials(QRSon_manual0, os.path.join(dataDir,'ToCSV','QRSon_q1c.csv'))
save_fiducials(QRSpeak_manual0, os.path.join(dataDir,'ToCSV','QRSpeak_q1c.csv'))
save_fiducials(QRSoff_manual0, os.path.join(dataDir,'ToCSV','QRSoff_q1c.csv'))
save_fiducials(Ton_manual0, os.path.join(dataDir,'ToCSV','Ton_q1c.csv'))
save_fiducials(Tpeak_manual0, os.path.join(dataDir,'ToCSV','Tpeak_q1c.csv'))
save_fiducials(Toff_manual0, os.path.join(dataDir,'ToCSV','Toff_q1c.csv'))

save_fiducials(Pon_manual1, os.path.join(dataDir,'ToCSV','Pon_q2c.csv'))
save_fiducials(Ppeak_manual1, os.path.join(dataDir,'ToCSV','Ppeak_q2c.csv'))
save_fiducials(Poff_manual1, os.path.join(dataDir,'ToCSV','Poff_q2c.csv'))
save_fiducials(QRSon_manual1, os.path.join(dataDir,'ToCSV','QRSon_q2c.csv'))
save_fiducials(QRSpeak_manual1, os.path.join(dataDir,'ToCSV','QRSpeak_q2c.csv'))
save_fiducials(QRSoff_manual1, os.path.join(dataDir,'ToCSV','QRSoff_q2c.csv'))
save_fiducials(Ton_manual1, os.path.join(dataDir,'ToCSV','Ton_q2c.csv'))
save_fiducials(Tpeak_manual1, os.path.join(dataDir,'ToCSV','Tpeak_q2c.csv'))
save_fiducials(Toff_manual1, os.path.join(dataDir,'ToCSV','Toff_q2c.csv'))
