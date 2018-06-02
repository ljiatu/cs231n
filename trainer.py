import copy
import time

import torch

MODEL_PATH = 'models/model.pt'
SCHEDULER_PATIENCE = 5


class Trainer:
    """
    Model trainer.
    """

    def __init__(
            self, model, loss_func, dtype, optimizer, device,
            loader_train, loader_val, loader_test, result_checker,
            num_epochs=10, print_every=50
    ):
        self.model = model
        self.loss_func = loss_func
        self.dtype = dtype
        self.optimizer = optimizer
        self.device = device
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.result_checker = result_checker
        self.num_epochs = num_epochs
        self.print_every = print_every

        # The place to run the scheduler during training phase depends on the scheduler itself,
        # so this dependency is a bit leaky. thus put it here.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE)

    def train(self):
        start = time.time()

        # Keep track of the best model.
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_acc = 0.0

        for e in range(self.num_epochs):
            print('-' * 10)
            print(f'Epoch {e}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for t, (x, y) in enumerate(self.loader_train):
                self.model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device).type(self.dtype)

                scores = self.model(x)
                loss = self.loss_func(scores, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Keep track of training loss throughout the epoch.
                training_loss = loss.item() * x.size(0)
                num_correct, num_samples = self.result_checker(scores, y)
                running_loss += training_loss
                running_corrects += num_correct
                total_samples += num_samples

                if t % self.print_every == 0:
                    print('Iteration %d, training loss = %.4f' % (t, loss.item()))
                    self._check_accuracy('train', self.loader_train)
                    self._check_accuracy('validation', self.loader_val)
                    print()

            epoch_training_loss = running_loss / total_samples
            epoch_training_acc = running_corrects.double() / total_samples
            epoch_val_loss, epoch_val_acc = self._check_accuracy('validation', self.loader_val)
            print('*' * 30)
            print(f'End of epoch {e} summary')
            print(f'Total samples: {total_samples}')
            print(f'Training loss: {epoch_training_loss}, accuracy: {epoch_training_acc * 100}%')
            print(f'Val loss: {epoch_val_loss}, accuracy: {epoch_val_acc * 100}%')
            print('*' * 30)

            self.scheduler.step(epoch_val_loss)
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self._save_model()

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(f'Best val accuracy: {best_val_acc * 100}%')
        self.model.load_state_dict(best_model_wts)

    def test(self):
        print('Test accuracy')
        self._check_accuracy('test', self.loader_test)

    def _check_accuracy(self, loader_label: str, loader) -> (float, float):
        total_num_correct = 0
        total_num_samples = 0
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device).type(self.dtype)
                scores = self.model(x)
                loss = self.loss_func(scores, y)
                total_loss += loss.item() * x.size(0)
                num_correct, num_samples = self.result_checker(scores, y)
                total_num_correct += num_correct
                total_num_samples += num_samples

            total_loss /= total_num_samples
            acc = float(total_num_correct) / total_num_samples
            print(f'{loader_label.capitalize()} Loss: {total_loss}')
            print(f'Got {total_num_correct} / {total_num_samples} correct ({acc * 100}%)')
            return total_loss, acc

    def _save_model(self):
        torch.save(self.model, MODEL_PATH)
