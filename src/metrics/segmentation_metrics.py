import torch
import torch.nn.functional as F

from utils.segmentation import multi_class_prediction, binary_prediction


class DiceScore():
    def __init__(self, nclasses, ignore_index=None, eps=1e-6):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.pred_fn = multi_class_prediction
        if nclasses == 1:
            self.nclasses += 1
            self.pred_fn = binary_prediction
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def calculate(self, output, target):
        batch_size = output.size(0)
        ious = torch.zeros(self.nclasses, batch_size)

        prediction = self.pred_fn(output)

        if self.ignore_index is not None:
            target_mask = (target == self.ignore_index).bool()
            prediction[target_mask] = self.ignore_index

        prediction = F.one_hot(prediction, self.nclasses).bool()
        target = F.one_hot(target, self.nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        total_count = (prediction.float() + target.float()).sum((-3, -2))
        ious = 2 * (intersection.float() + self.eps) / (total_count + self.eps)

        return ious.cpu()

    def update(self, value):
        self.mean_class += value.sum(0)
        self.sample_size += value.size(0)

    def value(self):
        return (self.mean_class / self.sample_size).mean()

    def reset(self):
        self.mean_class = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = self.mean_class / self.sample_size

        print(f'Dice Score: {self.value():.6f}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.6f}')

class MeanIoU():
    def __init__(self, nclasses, ignore_index=None, eps=1e-9):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def calculate(self, output, target):
        nclasses = output.size(1)
        prediction = torch.argmax(output, dim=1)
        prediction = F.one_hot(prediction, nclasses).bool()
        target = F.one_hot(target, nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        union = (prediction | target).sum((-3, -2))
        return intersection.cpu(), union.cpu()

    def update(self, value):
        self.intersection += value[0].sum(0)
        self.union += value[1].sum(0)
        self.sample_size += value[0].size(0)
        self.summary()

    def value(self):
        ious = (self.intersection + self.eps) / (self.union + self.eps)
        miou = ious.sum()
        nclasses = ious.size(0)
        if self.ignore_index is not None:
            miou -= ious[self.ignore_index]
            nclasses -= 1
        return miou / nclasses

    def reset(self):
        self.intersection = torch.zeros(self.nclasses).float()
        self.union = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = (self.intersection + self.eps) / (self.union + self.eps)

        print(f'mIoU: {self.value():.6f}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.6f}')


class PixelAccuracy():
    def __init__(self, nclasses, ignore_index=None):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.pred_fn = multi_class_prediction
        if nclasses == 1:
            self.nclasses += 1
            self.pred_fn = binary_prediction
        self.ignore_index = ignore_index
        self.reset()

    def calculate(self, output, target):
        prediction = self.pred_fn(output)

        image_size = target.size(1) * target.size(2)

        ignore_mask = torch.zeros(target.size()).bool().to(target.device)
        if self.ignore_index is not None:
            ignore_mask = (target == self.ignore_index).bool()
        ignore_size = ignore_mask.sum((1, 2))

        correct = ((prediction == target) | ignore_mask).sum((1, 2))
        acc = (correct - ignore_size + 1e-6) / \
            (image_size - ignore_size + 1e-6)
        return acc.cpu()

    def update(self, value):
        self.total_correct += value.sum(0)
        self.sample_size += value.size(0)

    def value(self):
        return (self.total_correct / self.sample_size).item()

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0

    def summary(self):
        print(f'Pixel Accuracy: {self.value():.6f}')
        
class _MeanIoU():
    def __init__(self, nclasses, ignore_index=None, eps=1e-9):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def calculate(self, output, target):
        nclasses = output.size(1)
        prediction = torch.argmax(output, dim=1)
        prediction = F.one_hot(prediction, nclasses).bool()
        target = F.one_hot(target, nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        union = (prediction | target).sum((-3, -2))
        ious = (intersection.float() + self.eps) / (union.float() + self.eps)
        return ious.cpu()

    def update(self, value):
        self.mean_class += value.sum(0)
        self.sample_size += value.size(0)

    def value(self):
        ious = self.mean_class
        miou = ious.sum() / self.sample_size
        nclasses = ious.size(0)
        if self.ignore_index is not None:
            miou -= ious[self.ignore_index] / self.sample_size
            nclasses -= 1
        return miou / nclasses

    def reset(self):
        self.mean_class = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = self.mean_class / self.sample_size

        print(f'mIoU: {self.value():.6f}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.6f}')


class ModifiedMeanIoU(MeanIoU):
    def calculate(self, output, target):
        return super().calculate(output[-1], target[-1])