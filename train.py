import os
import time
from functools import partial
import torch
from typing import Union, Optional, Text, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import abc


class Trainer(object):
    def name(self) -> Text:
        raise NotImplementedError

    def save(self, model: torch.nn.Module = None, optimizer=None, epoch: Optional[int] = None):
        """save model state dict into model path"""
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}
        path_checkpoint = f"models/{self.name()}-checkpoint_{epoch}_epoch.pkl"
        torch.save(checkpoint, path_checkpoint)


class TextClassifizerTrainer(Trainer):

    def name(self) -> Text:
        return "text_classifizer"

    def __init__(
            self, model: torch.nn.Module = None,
            args: Optional[Tuple] = None,
            train_dataloader: DataLoader = None,
            eval_dataloader: DataLoader = None,
            epochs: Optional[int] = 30,
            learning_rate: Optional[float] = 1e-5
    ):
        self.writer = SummaryWriter(
            f'logs/text-classifier-B-{train_dataloader.batch_size}-E{epochs}-L{learning_rate}-{time.time()}')
        self.writer.flush()

        if model is None:
            raise RuntimeError("`Trainer` requires a `model` ")
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args

    def train(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train_loop(epoch, loss_fn, optimizer)
            accu_val, loss = self.eval_loop(epoch, loss_fn)
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid accuracy {:8.3f} '
                  'valid loss {:8.3f} '
                  .format(epoch,
                          time.time() - epoch_start_time,
                          accu_val, loss))
            print('-' * 59)

    def train_loop(self, epoch, loss_fn, optimizer):
        self.model.train()
        total_acc, total_count = 0, 0

        for batch, data in enumerate(self.train_dataloader):
            y = data["labels"]
            X = data["text"]

            # Compute prediction and loss
            pred = self.model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_acc = (pred.argmax(1) == y).sum().item()
            current_count = y.size(0)
            loss, current = loss.item(), batch * len(X)

            total_acc += current_acc
            total_count += current_count

            # ...log the running loss
            self.writer.add_scalar('training loss',
                                   loss,
                                   (epoch - 1) * len(self.train_dataloader) + batch)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            # ...log the running loss
            self.writer.add_scalar('training acc',
                                   current_acc / current_count,
                                   (epoch - 1) * len(self.train_dataloader) + batch)

            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'
                  '| loss {:8.3f}'
                  .format(epoch, batch, len(self.train_dataloader),
                          current_acc / current_count, loss))

    def eval_loop(self, epoch, loss_fn):
        self.model.eval()
        total_acc, total_count = 0, 0
        loss = 0

        with torch.no_grad():
            for batch, data in enumerate(self.eval_dataloader):
                y = data["labels"]
                X = data["text"]

                pred = self.model(X)
                loss = loss_fn(pred, y)
                loss, current = loss.item(), batch * len(X)
                current_acc = (pred.argmax(1) == y).sum().item()
                current_count = y.size(0)

                total_acc += current_acc
                total_count += current_count

                # ...log the running loss
                self.writer.add_scalar('eval loss',
                                       loss,
                                       (epoch - 1) * len(self.eval_dataloader) + batch)

                self.writer.add_scalar('eval acc',
                                       current_acc / current_count,
                                       (epoch - 1) * len(self.eval_dataloader) + batch)

        return total_acc / total_count, loss


class SequenceLabelTrainer(Trainer):

    def name(self) -> Text:
        return "sequence-label"

    def __init__(
            self, model: torch.nn.Module = None,
            args: Optional[Tuple] = None,
            train_dataloader: DataLoader = None,
            eval_dataloader: DataLoader = None,
            epochs: Optional[int] = 30,
            learning_rate: Optional[float] = 1e-5
    ):
        self.writer = SummaryWriter(
            f'logs/sequence-label-B-{train_dataloader.batch_size}-E{epochs}-L{learning_rate}-{time.time()}')
        self.writer.flush()

        if model is None:
            raise RuntimeError("`Trainer` requires a `model` ")
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train_loop(epoch, optimizer)
            accu_val, loss = self.eval_loop(epoch)
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid accuracy {:8.3f} '
                  'valid loss {:8.3f} '
                  .format(epoch,
                          time.time() - epoch_start_time,
                          accu_val, loss))
            self.save(model=self.model, optimizer=optimizer, epoch=epoch)
            print('-' * 59)

    def train_loop(self, epoch, optimizer):
        self.model.train()
        for batch, data in enumerate(self.train_dataloader):
            y = data["labels"]
            X = data["text"]
            y_pred, padding_count = self.model(X)

            loss = self.model.loss(X, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_acc = (y_pred == y).sum().item() - padding_count
            current = y.size(0) * y.size(1) - padding_count

            loss = loss.item()

            # ...log the running loss
            self.writer.add_scalar('training loss',
                                   loss,
                                   (epoch - 1) * len(self.train_dataloader) + batch)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            # ...log the running loss
            self.writer.add_scalar('training acc',
                                   current_acc / current,
                                   (epoch - 1) * len(self.train_dataloader) + batch)

            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'
                  '| loss {:8.3f}'
                  .format(epoch, batch, len(self.train_dataloader),
                          current_acc / current, loss))

    def eval_loop(self, epoch):
        self.model.eval()
        total_acc, total_count = 0, 0
        loss = 0

        with torch.no_grad():
            for batch, data in enumerate(self.eval_dataloader):
                y = data["labels"]
                X = data["text"]

                y_pred, padding_count = self.model(X)

                loss = self.model.loss(X, y)
                current_acc = (y_pred == y).sum().item() - padding_count
                current = y.size(0) * y.size(1) - padding_count

                loss = loss.item()

                total_acc += current_acc
                total_count += current

                # ...log the running loss
                self.writer.add_scalar('eval loss',
                                       loss,
                                       (epoch - 1) * len(self.eval_dataloader) + batch)

                self.writer.add_scalar('eval acc',
                                       current_acc / current,
                                       (epoch - 1) * len(self.eval_dataloader) + batch)

        return total_acc / total_count, loss
