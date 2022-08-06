#coding:utf-8

from utils.parm import print_model_parm_flops, print_model_parm_nums

def simplesum(model, inputsize, device=-1):
    print_model_parm_nums(model)
    print_model_parm_flops(model, inputsize=inputsize, device=device)
