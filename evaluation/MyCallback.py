from tensorflow.python.keras.callbacks import Callback


class MyCallback(Callback):

    def __init__(self, val_input, val_target, train_input, train_target, batch_size):
        super().__init__()
        self._val_input = val_input
        self._val_target = val_target
        self._train_input = train_input
        self._train_target = train_target
        self._val_losses = []
        self._train_losses = []
        self._batch_size = batch_size

    def on_batch_end(self, batch, logs={}):
        val_loss = self.model.evaluate(self._val_input, self._val_target, verbose=0, batch_size=self._batch_size)
        train_loss = self.model.evaluate(self._train_input, self._train_target, verbose=0, batch_size=self._batch_size)
        self._val_losses.append(val_loss)
        self._train_losses.append(train_loss)
