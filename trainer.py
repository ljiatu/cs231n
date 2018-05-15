import copy

import time
import torch

MODEL_PATH = 'model/model.pt'


class Trainer:
    """
    Model trainer.
    """

    def __init__(
            self, model, loss_func, optimizer, scheduler, device,
            loader_train, loader_val, loader_test,
            num_epochs=10, print_every=50
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.num_epochs = num_epochs
        self.print_every = print_every

    def train(self):
        start = time.time()

        # Keep track of the best model.
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for e in range(self.num_epochs):
            print('-' * 10)
            print(f'\nEpoch {e}')
            print('-' * 10)

            self.scheduler.step()
            for t, (x, y) in enumerate(self.loader_train):
                self.model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = self.model(x)
                loss = self.loss_func(scores, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if t % self.print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item() * x.size(0)))
                    self._check_accuracy(self.loader_val)
                    print()

            print('*' * 10)
            print('Epoch accuracy at final iteration')
            print('*' * 10)
            epoch_acc = self._check_accuracy(self.loader_val)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self._save_model()

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(f'Best accuracy: {best_acc}')
        self.model.load_state_dict(best_model_wts)

    def test(self):
        print('Test accuracy')
        self._check_accuracy(self.loader_test)
        print()

    def _check_accuracy(self, loader):
        num_correct = 0
        num_samples = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                scores = self.model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            return acc

    def _save_model(self):
        torch.save(self.model, MODEL_PATH)
