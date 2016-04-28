import os, glob
import numpy as np
from PIL import Image
from collections import namedtuple
from random import shuffle
import shutil

caffe_root = '/lustre/yixi/segnet/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

sys.path.append('/lustre/yixi/face_segmentation_finetune/utils')
from convert_to_lmdb import *

if __name__=='__main__':
	if True:
		print 'convert to lmdb begins....\n'
		resize = True
		RSize = (480, 360)
		LabelSize = (480, 360)
		nopadding = True
		use_flow = []
		flow_dirs = ['flow_x', 'flow_y']
		RGB_mean_pad = False
		flow_mean_pad = True
		# Default is RGB_mean_pad = False and flow_mean_pad = True
		
		RGB_pad_values = [] if RGB_mean_pad else [0,0,0]
		flow_pad_value = 128 if flow_mean_pad else 0


		lmdb_dir = 'camvidtrainval' + ('rgbmp' if RGB_mean_pad else '') + ('fmp' if flow_mean_pad else '') + str(RSize[0]) + str(RSize[1]) + (''.join(use_flow)) + ('np' if nopadding else '') + '_lmdb'
			
		args = CArgs()
		args.resize = resize
		args.RSize = RSize
		args.LabelSize = LabelSize
		args.nopadding = nopadding
		args.use_flow = use_flow
		args.RGB_mean_pad = RGB_mean_pad
		args.flow_mean_pad =flow_mean_pad
		args.RGB_pad_values = RGB_pad_values
		args.flow_pad_value = flow_pad_value
		args.BoxSize = None # None is padding to the square of the longer edge
		args.NumLabels = 12 # [0,11]
		args.BackGroundLabel = 11
		args.lmdb_dir = lmdb_dir
		#args.proc_rank = proc_rank
		#args.proc_size = proc_size		

		#train_data = '/lustre/yixi/data/CamVid/701_StillsRaw_full/{id}.png'
		train_data = '/lustre/yixi/segnet/CamVid/train/{id}.png'
		val_data = '/lustre/yixi/segnet/CamVid/val/{id}.png'
		test_data = '/lustre/yixi/segnet/CamVid/test/{id}.png'
	 	train_label_data = '/lustre/yixi/segnet/CamVid/trainannot/{id}.png'
	 	val_label_data = '/lustre/yixi/segnet/CamVid/valannot/{id}.png'
	 	test_label_data = '/lustre/yixi/segnet/CamVid/testannot/{id}.png'
	 	flow_data = '/lustre/yixi/data/CamVid/flow_all/flow/{id}.{flow_type}.{flow_dir}.png'
		train_keys = [line.rstrip('\n') for line in open('/lustre/yixi/data/CamVid/p_train.txt')]
		val_keys = [line.rstrip('\n') for line in open('/lustre/yixi/data/CamVid/p_val.txt')]
		test_keys = [line.rstrip('\n') for line in open('/lustre/yixi/data/CamVid/p_test.txt')]

	
		inputs_all = [(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))]
		
		inputs_Train = dict([(k, train_data.format(id=k)) for k in train_keys])
		inputs_Val = dict([(k, val_data.format(id=k)) for k in val_keys])
		inputs_Test = dict([(k, test_data.format(id=k)) for k in test_keys])
		
		inputs_Train_Label = dict([(id, train_label_data.format(id=id)) for id in inputs_Train.keys()])	
		inputs_Val_Label = dict([(id, val_label_data.format(id=id)) for id in inputs_Val.keys()])
		inputs_Test_Label = dict([(id, test_label_data.format(id=id)) for id in inputs_Test.keys()])
		
		Train_keys = inputs_Train.keys()
		shuffle(Train_keys)
		Val_keys = inputs_Val.keys()
		shuffle(Val_keys)
		Test_keys = inputs_Test.keys()
		shuffle(Test_keys)

		flow_Train = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Train_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 
		flow_Val = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Val_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 
		flow_Test = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Test_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 


		inputs_TrainVal = inputs_Train.copy()
		inputs_TrainVal.update(inputs_Val)
		
		inputs_TrainVal_Label = inputs_Train_Label.copy()
		inputs_TrainVal_Label.update(inputs_Val_Label)
		
		flow_TrainVal = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Train_keys] + [(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Val_keys])  for flow_dir in flow_dirs for flow_type in use_flow] 
		
		TrainVal_keys = inputs_TrainVal.keys()
		shuffle(TrainVal_keys)


		if os.path.exists(lmdb_dir):
			shutil.rmtree(lmdb_dir, ignore_errors=True)

		if not os.path.exists(lmdb_dir):
			os.makedirs(lmdb_dir)

		############################# Creating LMDB for Training Data ##############################
		print("Creating Training Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'train-lmdb'), int(1e13), inputs_Train, flows=[],  keys=Train_keys, args=args)

		############################ Creating LMDB for Training Flow Data ##############################
		if flow_Train!=[]:
			print("Creating Training Flow LMDB File ..... ")
			createLMDBImage(os.path.join(lmdb_dir,'train-flow-lmdb'), int(1e13), None, flows=flow_Train,  keys=Train_keys, args=args)

		############################# Creating LMDB for Training Labels ##############################
		print("Creating Training Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'train-label-lmdb'), int(1e12), inputs_Train_Label, keys=Train_keys, args=args)
	



		############################# Creating LMDB for TrainVal Data ##############################
		print("Creating TrainVal Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'trainval-lmdb'), int(1e13), inputs_TrainVal, flows=[],  keys=TrainVal_keys, args=args)

		############################# Creating LMDB for TrainVal Flow Data ##############################
		if flow_TrainVal!=[]:
			print("Creating TrainVal Flow LMDB File ..... ")
			createLMDBImage(os.path.join(lmdb_dir,'trainval-flow-lmdb'), int(1e13), None, flows=flow_TrainVal,  keys=TrainVal_keys, args=args)
		 
		############################# Creating LMDB for TrainVal Labels ##############################
		print("Creating TrainVal Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'trainval-label-lmdb'), int(1e12), inputs_TrainVal_Label, keys=TrainVal_keys, args=args)




		############################# Creating LMDB for Validation Data ##############################
		print("Creating Validation Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'val-lmdb'), int(1e13), inputs_Val, flows=[],  keys=Val_keys, args=args)


		############################# Creating LMDB for Validation Flow Data ##############################
		if flow_Val!=[]:
			print("Creating Validation Data LMDB File ..... ")
			createLMDBImage(os.path.join(lmdb_dir,'val-flow-lmdb'), int(1e13), None, flows=flows_Val,  keys=Val_keys, args=args)

		############################# Creating LMDB for Validation Labels ##############################
		print("Creating Validation Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'val-label-lmdb'), int(1e12), inputs_Val_Label, keys=Val_keys, args=args)




		############################# Creating LMDB for Testing Data ##############################
		print("Creating Testing Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'test-lmdb'), int(1e13), inputs_Test, flows=[], keys=Test_keys, args=args)

		############################# Creating LMDB for Testing Flow Data ##############################
		if flow_Test!=[]:
			print("Creating Testing Data LMDB File ..... ")
			createLMDBImage(os.path.join(lmdb_dir,'test-flow-lmdb'), int(1e13), None, flows=flow_Test, keys=Test_keys, args=args)

		############################# Creating LMDB for Testing Labels ##############################
		print("Creating Testing Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'test-label-lmdb'), int(1e12), inputs_Test_Label, keys=Test_keys, args=args)

