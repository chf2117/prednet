from keras import losses
from cv2 import Laplacian
from cv2 import CV_8U

def blur_loss(y_true, y_pred):
    #weighting of blur loss
    alpha = 0.5
    mae = mean_absolute_error(y_true, y_pred)
    trueLap = Laplacian(y_true, CV_8U).var()
    predLap = Laplacian(y_pred, CV_8U).var()
    return mae + alpha * (trueLap - predLap) * (trueLap - predLap)
