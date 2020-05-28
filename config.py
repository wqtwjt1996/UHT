import argparse
import torch

def init_cfg():
    parser = argparse.ArgumentParser(description='Hyperparameters of UHT')
    parser.add_argument('--img_size', type=int, default=768,
                        help='size of the input image')
    parser.add_argument('--dataset_name', type=str, default='total_text',
                        choices=['total_text', 'coco_text', 'synth_text'],
                        help='size of the input image')
    parser.add_argument('--dataset_root', type=str, default='/home/andrew/Documents/Dataset/Total-Text',
                        help='path of dataset')
    parser.add_argument('--means', type=list, default=[0.485, 0.456, 0.406],
                        help='means of conversion of inputted images')
    parser.add_argument('--stds', type=list, default=[0.229, 0.224, 0.225],
                        help='standard deviations of conversion of inputted images')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch sizes of training')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether using GPU or not')
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='whether using more than 1 GPUs or not')
    parser.add_argument('--resume', type=str, default='/home/andrew/Documents/UHT/save/synth_text_resnet_50/model_0.pth',
                        help='resume path')
    parser.add_argument('--device', type=torch.device, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device type')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate of training')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch')
    parser.add_argument('--end_epoch', type=int, default=2000,
                        help='end epoch')
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='decay epoch when training')
    parser.add_argument('--decay_rate', type=float, default=0.8,
                        help='decay rate when training')
    parser.add_argument('--training_num_workers', type=int, default=4,
                        help='training num workers')
    parser.add_argument('--testing_num_workers', type=int, default=1,
                        help='testing num workers')
    parser.add_argument('--backbone', type=str, default='resnet_50',
                        choices=['resnet_50'],
                        help='backbone of UHT_Net')
    parser.add_argument('--save_frequency', type=int, default=25,
                        help='save checkpoint frequency')
    parser.add_argument('--visualization', type=bool, default=True,
                        help='whether visualization or not')
    parser.add_argument('--visualization_frequency', type=int, default=50,
                        help='visualization frequency')
    parser.add_argument('--save_directory', type=str, default="./save/",
                        help='checkpoint saving place')
    parser.add_argument('--prediction_output_directory', type=str, default="./output/",
                        help='prediction output saving place')
    parser.add_argument('--visualization_directory', type=str, default="./vis/",
                        help='visualization saving place')
    parser.add_argument('--evaluation_model_directory', type=str, default='/home/andrew/Documents/TextDetection_CVPR2020/save/total_text_resnet/model_300.pth',
                        help='model directory when evaluating the model')
    parser.add_argument('--textfill_top', type=float, default=0.7,
                        help='textfill algorithm hyperparameter')
    parser.add_argument('--textfill_flow_end', type=float, default=0.2,
                        help='textfill algorithm hyperparameter')
    parser.add_argument('--bounding_shape', type=bool, default=True,
                        help='final output text bounding box shape; True means bounding polygon; False means bounding quadrilateral')
    parser.add_argument('--spotter', type=bool, default=False,
                        help='whether adding ASTER as text recognition or not')
    parser.add_argument('--unit_disk', type=int, default=5,
                        help='variable: m in the pre-poccessing. Please refer to the paper for more details.')
    parser.add_argument('--max_annotation', type=int, default=2000,
                        help='')
    parser.add_argument('--max_points', type=int, default=20,
                        help='max points of text polygon')
    return parser.parse_args()
