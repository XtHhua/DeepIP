import gc

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


class ClassificationMetric(object):  # 记录结果并计算指标
    def __init__(
        self, accuracy=True, recall=True, precision=True, f1=True, average="macro"
    ):
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f1 = f1
        self.average = average

        self.preds = []
        self.target = []

    def reset(self):  # 重置结果
        self.preds.clear()
        self.target.clear()
        gc.collect()

    def update(self, preds, target):  # 更新结果
        preds = list(preds.cpu().detach().argmax(1).numpy())
        target = (
            list(target.cpu().detach().argmax(1).numpy())
            if target.dim() > 1
            else list(target.cpu().detach().numpy())
        )
        self.preds += preds
        self.target += target

    def compute(self):  # 计算结果
        metrics = []
        if self.accuracy:
            metrics.append(accuracy_score(self.target, self.preds))
        if self.recall:
            metrics.append(
                recall_score(
                    self.target,
                    self.preds,
                    labels=list(set(self.preds)),
                    average=self.average,
                )
            )
        if self.precision:
            metrics.append(
                precision_score(
                    self.target,
                    self.preds,
                    labels=list(set(self.preds)),
                    average=self.average,
                )
            )
        if self.f1:
            metrics.append(
                f1_score(
                    self.target,
                    self.preds,
                    labels=list(set(self.preds)),
                    average=self.average,
                )
            )
        self.reset()
        return metrics
