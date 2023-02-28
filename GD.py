from torch.nn import Module
from geopy.point import Point
from geopy.units import radians
import torch

def GD(test_label,perdict):

    test_label = test_label.reshape((len(test_label)*7,2))
    perdict = perdict.reshape((len(perdict)*7,2))

    test_label = torch.tensor(test_label,dtype=torch.float64)
    perdict = torch.tensor(perdict, dtype=torch.float64)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#cuda
    criterion = GDLoss_().to(device)

    loss = criterion(perdict,test_label,device)
    return loss*6371.009

class GDLoss_(Module):
    def __init__(self):
        super(GDLoss_, self).__init__()
        self.pred = None
        self.true = None

    def forward(self, pred, true,device):
        self.pred = pred.clone()
        self.true = true.clone()

        b, m = pred.shape
        sum_gd = torch.zeros(1).to(device=device)
        for i in range(b):
            self._range(i)
            temp_gd = self.measure(self.pred[i, ...], self.true[i, ...])
            sum_gd += temp_gd

        # return sum_gd
        return sum_gd / b

    def _range(self, i):
        if self.pred[i, 0] < - 90:
            self.pred[i, 0] = -90
        elif self.pred[i, 0] > 90:
            self.pred[i, 0] = 90

    def measure(self, a, b):
        a, b = Point(a), Point(b)

        lat1, lng1 = torch.tensor(radians(degrees=a.latitude)), torch.tensor(radians(degrees=a.longitude))
        lat2, lng2 = torch.tensor(radians(degrees=b.latitude)), torch.tensor(radians(degrees=b.longitude))

        sin_lat1, cos_lat1 = torch.sin(lat1), torch.cos(lat1)
        sin_lat2, cos_lat2 = torch.sin(lat2), torch.cos(lat2)

        delta_lng = lng2 - lng1
        cos_delta_lng, sin_delta_lng = torch.cos(delta_lng), torch.sin(delta_lng)

        d = torch.atan2(torch.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                                   (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

        # return self.RADIUS * d
        return d



