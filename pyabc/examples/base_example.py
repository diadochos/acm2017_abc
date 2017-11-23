import inspect
import abc

class Example(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def simulator(self):
        pass

    @property
    def summaries(self):
        return self._summaries()

    @abc.abstractmethod
    def _summaries(self):
        pass

    @property
    def summary_names(self):
        return [f.__name__ for f in self._summaries()]


    @property
    def params(self):
        return inspect.getargspec(self.simulator)[0][1:]

    @property
    def nr_params(self):
        return len(self.params)
