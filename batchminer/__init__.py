from batchminer import random, semihard, distance

BATCHMINING_METHODS = {'random':random,
                       'semihard':semihard,
                       'distance':distance,
                       }

def select(name, **kwargs):
    #####
    if name not in BATCHMINING_METHODS: raise NotImplementedError('Batchmining {} not available!'.format(name))

    batchmine_lib = BATCHMINING_METHODS[name]

    return batchmine_lib.BatchMiner(**kwargs)
