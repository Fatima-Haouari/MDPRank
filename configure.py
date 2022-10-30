import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
# print(tf.__version__)

# import tensorflow.compat.v1 as tensorflow
# tf.disable_v2_behavior()
from pyterrier.measures import RR, R, Rprec, P, MAP, nDCG
from tensorflow import flags




flags.DEFINE_list("eval_metrics", [MAP@5, P@1, P@5, R@5, R@50, nDCG@5], "Evaluation metrics")

flags.DEFINE_string('data','AuFin','data set')
flags.DEFINE_float("dropout_keep_prob",0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("learning_rate", 0.00001, "learn rate( default: 0.0)")

flags.DEFINE_integer("feature_dim", 768, "feature size")
flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
flags.DEFINE_string("file_name","Fold2","current_file_name")

# authority finder data paths :
# train: /data/authority-finder/AuFIN_RL/Fold1/train.txt
# validation: /data/authority-finder/AuFIN_RL/Fold1/dev.txt
# test: /data/authority-finder/AuFIN_RL/Fold1/test.txt
# flags.DEFINE_string('train_data','/data/authority-finder/AuFIN_RL/Fold1/train.txt',' train_data')
# flags.DEFINE_string('val_data','/data/authority-finder/AuFIN_RL/Fold1/dev.txt','val_data') 
# flags.DEFINE_string('test_data', '/data/authority-finder/AuFIN_RL/Fold1/test.txt','test_data') 

flags.DEFINE_string('train_data','/data/authority-finder/AuFIN_RL/Fold2/train.txt',' train_data')
flags.DEFINE_string('val_data','/data/authority-finder/AuFIN_RL/Fold2/dev.txt','val_data') 
flags.DEFINE_string('test_data', '/data/authority-finder/AuFIN_RL/Fold2/test.txt','test_data') 

flags.DEFINE_string('QRELS', './pre_prosess/AuFIN/qrels_150claims_final.txt','gold labels')


# verified claimm retrieval data paths:
# train_output = './pre_prosess/VCR22/en-2022-train-formatted.tsv'
# val_output = './pre_prosess/VCR22/en-2022-validation-formatted.tsv'
# test_output = './pre_prosess/VCR22/en-2022-test-formatted.tsv'
# vc_qrels = pre_prosess/VCR22/All_QRELs.tsv

# flags.DEFINE_integer("feature_dim", 50, "feature size")
# flags.DEFINE_string('QRELS', './pre_prosess/VCR22/All_QRELs.tsv','gold labels')
# flags.DEFINE_string('train_data','./pre_prosess/VCR22/en-2022-train-formatted.tsv',' train_data')
# flags.DEFINE_string('val_data','./pre_prosess/VCR22/en-2022-validation-formatted.tsv','val_data') # to be filled properly
# flags.DEFINE_string('test_data', './pre_prosess/VCR22/en-2022-test-formatted.tsv','test_data') # to be filled properly

# flags.DEFINE_string("file_name","Fold1","current_file_name")
# Training parameters
# flags.DEFINE_integer("batch_size", 320, "Batch Size (OHSUMED dataset)")

flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

flags.DEFINE_float('reward_decay',0.99,'reward_decay')


# flags.DEFINE_string('CNN_type','ircnn','data set')

flags.DEFINE_float('sample_train',1,'sampe my train data')
flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
# flags.DEFINE_string('pooling','max','pooling strategy')
flags.DEFINE_boolean('clean',False,'whether clean the data')
# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#data_helper para

# flags.DEFINE_boolean('isEnglish',True,'whether data is english')
