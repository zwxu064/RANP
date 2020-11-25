eps = 1e-5
def f1_score(o, t):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


#https://github.com/ellisdg/3DUnetCNN
#https://github.com/ellisdg/3DUnetCNN/blob/master/brats/evaluate.py
#https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py
def dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0
    ret += f1_score(o, t),
    # core
    o = (output==1) | (output==4)
    t = (target==1) | (target==4)
    ret += f1_score(o , t),
    # active
    o = (output==4); t = (target==4)
    ret += f1_score(o , t),

    return ret

keys = 'whole', 'core', 'enhancing', 'loss'