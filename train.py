#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
# from pre_dateset import batch_gen_with_point_wise,load,prepare,batch_gen_with_single
from pre_process import load_data, get_batch, get_each_query_length, get_batch_with_test
from model_new import QRL_L2R
# from evaluation import evaluation_ranklists
from evaluation import evaluate_trec_run
import random
# import evaluation as evaluation_test
# import evaluation_test
# import cPickle as pickle
import pickle
# from sklearn.model_selection import train_test_split
import configure
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
import pandas as pd
import global_variables as gb


set_random_seed(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from functools import wraps

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()
# FLAGS._parse_flags()

now = int(time.time()) 
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)

log_dir = 'log/'+ timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
para_file = log_dir + '/test_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'

def predict(RL_L2R, dataset, run_save_path = ""):

	reward_sum = 0
	label_collection = []
	df_trec = pd.DataFrame(columns = [gb.QID, gb.Q0, gb.DOC_NO, gb.RANK, gb.SCORE, gb.TAG])

	for query_data in get_batch_with_test(dataset, FLAGS.feature_dim):
		query_docs_features = query_data[0]
		query_docs_labels = query_data[1]
		query_docs_ids = query_data[2]
		query_docs_count = query_data[3]
		qid = query_data[4]

		for step in range(query_docs_count):
			immediate_rewards = calcu_immediate_reward(step, query_docs_labels)
	
			selected_doc_index = RL_L2R.choose_doc(step, query_docs_features, query_docs_labels, immediate_rewards)
				
			# selected_doc_index = RL_L2R.choose_doc(step,doc_feature)
			current_doc = query_docs_features[selected_doc_index]
			current_label = query_docs_labels[selected_doc_index]
			current_doc_id = query_docs_ids[selected_doc_index][0]
			query_docs_features, query_docs_labels = exclude_selected_doc(selected_doc_index, query_docs_features, query_docs_labels) 
			# reward = calcu_reward(step, current_label)
			# RL_L2R.store_transition(current_doc,current_label,reward)			
			RL_L2R.store_transition(current_doc, current_label, current_doc_id)

		reward = calcu_return(RL_L2R.query_docs_labels)
		label_collection.append(RL_L2R.query_docs_labels)
		reward_sum += reward
		df_trec = add_new_query(df_trec, qid, RL_L2R.query_docs_ids)
		RL_L2R.reset_query_transitions()

	# save the run
	if run_save_path != "":
		if not os.path.isfile(run_save_path):
			os.makedirs(os.path.dirname(run_save_path), exist_ok=True)
		df_trec.to_csv(run_save_path, index=False, sep='\t', encoding="utf-8")
	print("type(reward_sum)"),type(reward_sum))
	print("(reward_sum)",reward_sum)
	return label_collection, reward_sum, df_trec




def add_new_query(df_trec, query_id, doc_ids):
	df_query = pd.DataFrame(columns = [gb.QID, gb.Q0, gb.DOC_NO, gb.RANK, gb.SCORE, gb.TAG])
	df_query[gb.DOC_NO] = doc_ids
	df_query[gb.QID] = [query_id] * len(doc_ids)
	df_query[gb.SCORE] = [1] * len(doc_ids)
	df_query[gb.Q0] = [gb.Q0] * len(doc_ids)
	df_query[gb.RANK] = [(i+1) for i in range(len((doc_ids)))]
	df_query[gb.TAG] = ["RL_run"] * len(doc_ids)
	df_trec = df_trec.append(df_query, ignore_index=True)
	return df_trec


# def get_candidata_feature_label(index, doc_feature, doc_label):
# 	r_doc_feature = np.delete(doc_feature, index, 0)
# 	r_doc_label = np.delete(doc_label, index, 0)
# 	return r_doc_feature, r_doc_label
def exclude_selected_doc(index, doc_feature, doc_label):
	r_doc_feature = np.delete(doc_feature, index, 0)
	r_doc_label = np.delete(doc_label, index, 0)
	return r_doc_feature, r_doc_label

def calcu_return(query_docs_labels):
	# discounted_ep_rs = np.zeros_like(ep_rs)
	running_add = 0
	DCG = calcu_DCG(query_docs_labels) 
	for t in range(len(query_docs_labels)):
		running_add += (FLAGS.reward_decay**t) * DCG[t]
		# discounted_ep_rs[t] = runing_add

	return running_add

def calcu_DCG(query_docs_labels):
	DCG = []
	for i in range(len(query_docs_labels)):
		if i == 0:
			DCG.append(np.power(2.0, query_docs_labels[i]) - 1.0)
		else:
			DCG.append((np.power(2.0, query_docs_labels[i]) - 1.0)/np.log2(i+1))
	# print ("DCG : {}".format(DCG))
	return DCG

def calcu_immediate_reward(current_step, labels):
	decay_rate = FLAGS.reward_decay
	DCG = []
	# print (decay_rate**current_step)
	if current_step == 0:
		for i in range(len(labels)):
			temp_DCG = np.power(2.0, labels[i]) - 1.0
			DCG.append((decay_rate**current_step)*temp_DCG)
	else:
		for i in range(len(labels)):
			temp_DCG = (np.power(2.0, labels[i]) - 1.0)/np.log2(current_step+1)
			DCG.append((decay_rate**current_step)*temp_DCG)

	return DCG


def get_result_message(result_type, res, epoch, reward):

	result_message = "## epoch {},".format(epoch,)
	for metric in res:
		result_message += " , " + result_type + "_" + metric + " : " + str(round(res[metric], 4))
	result_message += "\n" +  result_type + " _reward : {}".format(round(reward, 4))
	return result_message


def train():
	file_name = FLAGS.file_name
	train_set = load_data(FLAGS.train_data)
	valid_set = load_data(FLAGS.val_data)
	test_set =  load_data(FLAGS.test_data)
	# train_set = load_data("./pre_prosess/OHSUMED/"+file_name+"/trainingset.txt")
	# test_set = load_data("./pre_prosess/OHSUMED/"+file_name+"/testset.txt")
	# valid_set = load_data("./pre_prosess/OHSUMED/"+file_name+"/validationset.txt")
	each_query_length = get_each_query_length()

	log = open(precision,"w")
	log.write(str(FLAGS.__flags)+'\n')

	RL_L2R = QRL_L2R(
			feature_dim = FLAGS.feature_dim,
			learn_rate = FLAGS.learning_rate,
			reward_decay = FLAGS.reward_decay,
			FLAGS = FLAGS
			)
			
	max_ndcg_1 = 0.020
	max_ndcg_10 = 0.02
	max_reward = 1
	# loss_max = 0.3
	for i in range(FLAGS.num_epochs):
		print ("\nepoch "+str(i)+"\n")
		j = 1
		# reward_sum = 0
		# training process
		for query_data in get_batch(train_set, FLAGS.feature_dim): #for each query
			query_docs_features = query_data[0] #Xn features vectors
			query_docs_labels = query_data[1] #Yn
			# print("len(docs_labels)",len(query_docs_labels))
			query_docs_ids = query_data[2] # Xn documents IDs
			query_docs_count = query_data[3] #number of documents associated with the query
			# print("len(docs_count)",query_docs_count)
			qid = query_data[4]
			# print ("doc_label : {}".format(doc_label))
			for step in range(query_docs_count):
				immediate_rewards = calcu_immediate_reward(step, query_docs_labels) #R: Algorithm 1 reward function
				# print(immediate_rewards)
				# print(len(immediate_rewards))
				selected_doc_index = RL_L2R.choose_doc(step, query_docs_features, query_docs_labels, immediate_rewards, True)
				current_doc = query_docs_features[selected_doc_index]
				current_label = query_docs_labels[selected_doc_index]
				current_doc_id = query_docs_ids[selected_doc_index]
				query_docs_features, query_docs_labels = exclude_selected_doc(selected_doc_index, query_docs_features, query_docs_labels) 
				# print (current_label)
				RL_L2R.store_transition(current_doc,current_label, current_doc_id)

			# print ("RL_L2R.ep_label : {}".format(RL_L2R.ep_label))
			query_return = calcu_return(RL_L2R.query_docs_labels)
			# print (query_return)
			# idel_reward, idel_features = calcu_idel_reward(RL_L2R.ep_docs, RL_L2R.ep_label)
			# ep_rs_norm, loss = RL_L2R.learn(query_return)		
			# ep_rs_norm, loss = RL_L2R.learn(query_return, idel_reward, idel_features)
			# loss = RL_L2R.learn(query_return)
			RL_L2R.reset_query_transitions()
			# reward_sum += query_return
			print ("training, qid :{} with_length : {}, reward : {}".format(qid, query_docs_count, query_return))
			# break

		# train evaluation
		train_predict_label_collection, train_reward, df_train_trec = predict(RL_L2R, train_set)	
		train_res = evaluate_trec_run(df_train_trec)
		train_result_line = get_result_message("train", train_res, epoch=i, reward=train_reward[0])
		# train_result_line = "## epoch {}, train MAP@5 : {}, train_ P@1 : {}, train_P@5 : {}, train_ R@5 : {}, train_ R@50 : {}, \ntrain_reward : {}".format(i, train_MAP, train_P1, train_P5, train_R5, train_R50,  reward=train_reward[0])

		print (train_result_line)		
		log.write(train_result_line+"\n")	


		
		# valid evaluation
		valid_predict_label_collection, valid_reward, df_dev_trec = predict(RL_L2R, valid_set)	
		val_res = evaluate_trec_run(df_dev_trec)
		valid_result_line = get_result_message("validation", val_res, epoch=i, reward=valid_reward[0])

		# valid_MAP, valid_P1, valid_P5, valid_R5, valid_R50  = evaluate_trec_run(df_dev_trec)
		# valid_result_line = "## epoch {}, valid MAP@5 : {}, valid_ P@1 : {}, valid_P@5 : {}, valid_ R@5 : {}, valid_ R@50 : {}, \nvalid_reward : {}".format(i, 
		# valid_MAP, valid_P1, valid_P5, valid_R5, valid_R50, valid_reward[0])

		print (valid_result_line)
		log.write(valid_result_line+"\n")

		# save param	
		# FIXME
		if valid_reward > max_reward: #total return 
			max_reward = valid_reward[0] 
			write_str = str(max_reward) +"_P@1_"+str(val_res['P@1'])
			RL_L2R.save_param(write_str, timeDay)


		# if valid_NDCG_at_1 > max_ndcg_1 and valid_NDCG_at_10 > max_ndcg_10:
		# 	max_ndcg_1 = valid_NDCG_at_1
		# 	max_ndcg_10 = valid_NDCG_at_10
		# 	write_str = str(max_ndcg_1)+"_"+str(max_ndcg_10)
		# 	RL_L2R.save_param(write_str, timeDay)

		# test evaluation
		test_predict_label_collection, test_reward, df_test_trec = predict(RL_L2R, test_set)

		test_res = evaluate_trec_run(df_test_trec)
		test_result_line = get_result_message("test", test_res, epoch=i, reward=test_reward[0])

		# test_MAP, test_P1, test_P5, test_R5, test_R50  = evaluate_trec_run(df_test_trec)
		# test_result_line = "## epoch {}, test MAP@5 : {}, test_ P@1 : {}, test_P@5 : {}, test_ R@5 : {}, test_ R@50 : {}, \ntest_reward : {}".format(i, test_MAP,
		# 		 test_P1, test_P5, test_R5, test_R50, test_reward[0])

		print (test_result_line)
		log.write(test_result_line+"\n\n")


	# test process
	
	test_predict_label_collection, test_reward, df_test_trec = predict(RL_L2R, test_set)				
	test_res = evaluate_trec_run(df_test_trec)
	test_result_line = get_result_message("test", test_res, epoch='final', reward=test_reward[0])

	# test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR, test_P = evaluate_trec_run(test_predict_label_collection)
	# test_result_line = "## test_MAP : {}, test_NDCG_at_1 : {}, test_NDCG_at_3 : {}, test_NDCG_at_5 : {}, test_NDCG_at_10 : {}, test_NDCG_at_20 : {}, test_MRR@20 : {}, test_P@20 : {}, \ntest_reward : {}".format(test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR, test_P, test_reward[0])
	print (test_result_line)
	log.write(test_result_line+"\n")

	log.write("\n")
	log.flush()
	log.close()
	
	# label_collection = []
	# for data in get_batch_with_test(test_set):
	# 	doc_feature = data[0]
	# 	doc_label = data[1]
	# 	doc_len = data[2]
	# 	for step in range(doc_len):
	# 		selected_doc_index = RL_L2R.choose_doc(step,doc_feature)
	# 		current_doc = doc_feature[selected_doc_index]
	# 		current_label = doc_label[selected_doc_index]
	# 		doc_feature, doc_label = get_candidata_feature_label(selected_doc_index, doc_feature, doc_label) 
	# 		reward = calcu_reward(step, current_label)
	# 		RL_L2R.store_transition(current_doc,current_label,reward)
	# 	label_collection.append(RL_L2R.ep_lable)
			

			# if j == 5: 
			# 	exit()
			# j+=1
	# line1 = " {}:epoch: map_train{}".format(i,map_NDCG0_NDCG1_ERR_p_train)
	# log.write(line1+"\n")
	# line = " {}:epoch: map_test{}".format(i,map_NDCG0_NDCG1_ERR_p_test)
	# log.write(line+"\n")
	# log.write("\n")
	# log.flush()
	# log.close()
	# exit()


if __name__ == "__main__":
	train()
			