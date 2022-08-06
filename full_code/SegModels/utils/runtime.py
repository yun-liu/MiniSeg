#coding:utf-8
import time
import numpy as np
import torch
from torch.autograd import Variable


def runtime(model, inputsize, iter=100, device=-1):

    model.eval()
    time_spent = []

    if device >= 0 and torch.cuda.is_available():
        input = Variable(torch.rand(inputsize).unsqueeze(0), requires_grad=False).cuda(device)
    else:
        input = Variable(torch.rand(inputsize).unsqueeze(0), requires_grad=False)

    for idx in range(iter):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        if idx >= 10:
            time_spent.append(time_taken)
    print("Run-Time: %.4f s" % (np.mean(time_spent)))
