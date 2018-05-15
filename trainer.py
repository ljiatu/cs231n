import torch

MODEL_PATH = 'model/model.pt'


class Trainer:
    """
    Model trainer.
    """

    def __init__(
            self, model, loss_func, optimizer, device,
            loader_train, loader_val, loader_test,
            num_epochs=10, print_every=50
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.num_epochs = num_epochs
        self.print_every = print_every

    def train(self):
        for e in range(self.num_epochs):
            print(f'\nEpoch {e}')
            # Save the model at the beginning of each epoch.
            self._save_model()
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
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    self._check_accuracy(self.loader_val)
                    print()

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

    def _save_model(self):
        torch.save(self.model, MODEL_PATH)
