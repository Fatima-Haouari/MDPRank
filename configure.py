import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
# print(tf.__version__)

# import tensorflow.compat.v1 as tensorflow
# tf.disable_v2_behavior()
from pyterrier.measures import RR, R, Rprec, P, MAP
from tensorflow import flags




flags.DEFINE_list("eval_metrics", [MAP@5, P@1, P@5, R@5, R@50,], "Evaluation metrics")

flags.DEFINE_float("dropout_keep_prob",0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("learning_rate", 0.00001, "learn rate( default: 0.0)")

flags.DEFINE_string('QRELS', './pre_prosess/AuFIN/qrels_150claims_final.txt','gold labels')
flags.DEFINE_string('train_data','./pre_prosess/AuFIN/Fold1/train_Biolists_iter1_tune.txt',' train_data')
flags.DEFINE_string('val_data','val_data','val_data') # to be filled properly
flags.DEFINE_string('test_data', 'test_data','test_data') # to be filled properly

flags.DEFINE_string('data','OHSUMED','data set')
# flags.DEFINE_string("file_name","Fold1","current_file_name")
flags.DEFINE_string("file_name","Fold5","current_file_name")
# Training parameters
# flags.DEFINE_integer("batch_size", 320, "Batch Size (OHSUMED dataset)")
flags.DEFINE_integer("feature_dim", 768, "feature size")

flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")

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
