import numpy as np

class MinMaxScaler():
    def __init__(self, scale_min, scale_max):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.origin_min = None
        self.origin_max = None
    
    def fit_transform(self, x):
        self.origin_min = x.min()
        self.origin_max = x.max()
        x = (x - self.origin_min) / (self.origin_max - self.origin_min)
        x = (self.scale_max - self.scale_min) * x + self.scale_min
        return x
    
    def inverse_transform(self, x):
        x = (x - self.scale_min) / (self.scale_max - self.scale_min)
        x = x * (self.origin_max - self.origin_min) + self.origin_min
        return x

def main():
    scaler = MinMaxScaler(-1, 1)
    test = np.arange(20).reshape(2, 2, 5)
    t = scaler.fit_transform(test)
    tt = scaler.inverse_transform(t)
    print(t)
    print(tt)
    
if __name__ == '__main__':
    main()
