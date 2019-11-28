def flatten(x):
    N = x.shape[0]
    return x.view(N,-1)