"""Module for classification using quantum machine learning models.
"""
import numpy as np

import pickle
import json
from tqdm.auto import tqdm

from scipy.optimize import minimize

from .quantumModel import quantumModel
from .optim import OPTIM_METHODS


from .utility import CE_loss, CE_grad

SCIPY_METHODS = {
    'bfgs', 'nelder-mead', 'powell', 'cg',
    'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla',
    'slsqp', 'trust-constr', 'dogleg',
}
GRAD_BASED = {
    'cg', 'bfgs', 'newton-cg', 'l-bfgs-b',
    'tnc', 'slsqp', 'dogleg', 'trust-ncg',
    'trust-krylov', 'trust-exact', 'trust-constr'
}


class Classifier(quantumModel):
    """Class for quantum classifiers. Defines the API using the scikit-learn
    format.

    Parameters
    ----------
    circuit : circuitML
        Quantum circuit to simulate, how to use and store is defined in child
        classes.
    bitstr : list of int or list of str
        Which bitstrings should correspond to each class. The number of
        classes for the classification is defined by the number of elements.
    params : vector, optional
        Initial model paramters. If ``None`` (default) uses
        :meth:`circuitML.random_params`.
    nbshots : int, optional
        Number of shots for the quantum circuit. If 0, negative or None, then
        exact proabilities are computed, by default ``None``.
    nbshots_increment : float, int or callable, optional
        How to increase the number of shots as optimization progress. If float
        or int, the increment arise every `nbshots_incr_delay` iterations: if
        float, then the increment is multiplicative; if int, then it is added.
        If callable, the new nbshots is computed by calling
        `nbshots_increment(nbshots, n_iter, loss_value)`.
    nbshots_incr_delay : int, optional
        After how many iteration nb_shots has to increse. By default 20, if
        nbshots_increment is given
    loss : callable, optional
        Loss function, by default Negative LogLoss (Cross entropy).
    job_size : int, optional
        Number of runs for each circuit job, by default the number of
        observations.
    budget : int, optional
        Maximum number of optimization steps, by default 100
    name : srt, optional
        Name to identify this classifier.
    save_path : str, optional
        Where to save intermediate training results, by deafult None. If
        ``None``, intermediate results are not saved.

    Attributes
    ----------
    bitstr : list[int]
        Bitstrings (as int) on which to read the classes
    nbshots : int
        Number of shots to run circuit
    job_size : int
        Number of circuits to run in each backend job
    nfev : int
        Number if times the circuit has been run
    """
    def __init__(self, circuit, bitstr, **kwargs):
        super().__init__(circuit, **kwargs)

        # Retrieve keyword arguments
        loss = kwargs.get('loss', CE_loss)
        loss_grad = kwargs.get(
            'loss_grad',
            CE_grad if loss is CE_loss else None
        )
        name = kwargs.get('name')
        save_path = kwargs.get('save_path')

        # Setting bitstrings
        self.set_bitstr(bitstr)

        self.__loss__ = loss
        self.__loss_grad__ = loss_grad
        self.__min_loss__ = np.inf
        self.__fit_conv__ = False

        self.__last_loss_value__ = None
        self.__last_output__ = None
        self.__last_params__ = None
        self.__loss_progress__ = []
        self.__output_progress__ = []
        self.__params_progress__ = []
        self.__name__ = name
        self.__save_path__ = save_path

    def set_bitstr(self, bitstr):
        """Bitstring setter

        Parameters
        ----------
        bitstr : list[str] or list[int]
            Bitstrings on which to read the class predictions.

        Raises
        ------
        TypeError
            If bitstrings are of wrong type or have eterogenous types
        """
        if isinstance(bitstr[0], int):
            for i in bitstr:
                if not isinstance(i, int):
                    raise TypeError("All bitstrings must have the same type")
            self.bitstr = bitstr
        elif isinstance(bitstr[0], str):
            for i in bitstr:
                if not isinstance(i, str):
                    raise TypeError("All bitstrings must have the same type")
            self.bitstr = [int(bit, 2) for bit in bitstr]
        else:
            raise TypeError("Bitstrings must be either int or binary strings")

    def predict_proba(self, X, params=None):
        """Compute the bitstring probabilities associated to each input point
        of the design matrix.

        Parameters
        ----------
        X : array
            Design matrix of n samples
        params : vector, optional
            Circuit parameters, by default None. If not given, model
            parameters are used.

        Returns
        -------
        array
            Predicted bitstring probabilities. Rows correspond to samples and
            columns to bitstrings, whose order is defined in
            :attr:`~polyadicqml.quantumClassifier.bitstr`.
        """
        out = self.run_circuit(X, params)

        if self.nbshots:
            out = out / float(self.nbshots)

        return out[:, self.bitstr]

    def grad(self, X, y, params=None):
        # Compute the Jacobian vector product of
        # Jac_{params}(circuit) @ Grad_{bistr}(loss)
        if params is None:
            params = self.params

        if len(X.shape) < 2:
            X = X.reshape(1, -1)
        N, *_ = X.shape

        v = np.zeros((N, 2**self.circuit.nbqbits))
        v[:, self.bitstr] = self.__loss_grad__(
            y, self.predict_proba(X, params)
        )[:, :, 0]

        return self.circuit.grad(
            X, params,
            v=v,
            nbshots=self.nbshots,
            job_size=self.job_size,
        ).flatten()

    def proba_to_label(self, proba) -> np.ndarray:
        """Transforms a matrix of real values in integer labels.

        Parameters
        ----------
        proba : array
            Real valued array

        Returns
        -------
        vector
            Labels vector
        """
        return np.argmax(proba, axis=1)

    def predict(self, X):
        """Compute the predicted class for each input point of the design
        matrix.

        Parameters
        ----------
        X : array
            Design matrix of n samples

        Returns
        -------
        vector
            Labels vector
        """
        return self.proba_to_label(self.predict_proba(X))

    def set_loss(self, loss=None):
        """Loss function setter.

        Parameters
        ----------
        loss : callable, optional
            Loss function of the form loss(y_true, y_pred, labels), by default
            None. If None is given, nothing happens.
        """
        if loss is not None:
            self.__loss__ = loss

    def __callback__(self, params, loss=False, output=False, ):
        """Callback function for optimization. It is called after each step.

        Parameters
        ----------
        params : vector
            Current parameter vector
        loss : bool, optional
            Wheter to store the loss value, by default False
        output : bool, optional
            Wheter to store the current output and parameters , by default
            False
        """
        self.__n_iter__ += 1
        self.pbar.update()

        if loss or output:
            self.__loss_progress__.append(self.__last_loss_value__)
        if output:
            self.__output_progress__.append(self.__last_output__.tolist())
            self.__params_progress__.append(params.tolist())

        if self.__save_path__ and self.__n_iter__ % 10 == 0:
            self.save()

        # We randomize the indices only after the callback
        # this is necessary to estimate the gradient by FD
        self._rnd_indices = np.random.choice(
            self.__indices, size=self.__batch_size, replace=False)

    def __scipy_minimize__(
            self, input_train, target_train, labels,
            **kwargs
    ):
        method = kwargs.get('method', 'BFGS').lower()
        save_loss_progress = kwargs.get('save_loss_progress', None)
        save_output_progress = kwargs.get('save_output_progress', None)

        def to_optimize(params):
            self.nbshots = self.nbshots_increment(
                self.nbshots, self.__n_iter__, self.__min_loss__)

            probas = self.predict_proba(
                input_train[self.__rnd_indices], params
            )
            loss_value = self.__loss__(
                target_train[self.__rnd_indices], probas, labels=labels
            )

            self.__last_loss_value__ = loss_value
            self.__last_output__ = probas[np.argsort(self.__rnd_indices)]

            if loss_value < self.__min_loss__:
                self.__min_loss__ = loss_value
                self.set_params(params.copy())

            if method == "cobyla":
                self.__callback__(
                    params, save_loss_progress, save_output_progress
                )

            return loss_value

        # SCIPY.MINIMIZE IMPLEMENTATION
        options = kwargs.get('options', {'maxiter': self.__budget__})
        bounds = kwargs.get('bounds')
        if method == 'L-BFGS-B' and bounds is None:
            bounds = [(-np.pi, np.pi) for _ in self.params]

        mini_kwargs = dict(
            method=method,
            bounds=bounds,
            options=options,
        )
        if method not in ('cobyla'):
            mini_kwargs["callback"] = lambda xk: self.__callback__(
                xk, save_loss_progress, save_output_progress,
            )
        if method in GRAD_BASED:
            def jac(params):
                return self.grad(input_train, target_train, params)

            mini_kwargs['jac'] = jac

        mini_out = minimize(to_optimize, self.params, **mini_kwargs)

        self.set_params(mini_out.x.copy())
        self.__fit_conv__ = mini_out.success

    def __optim__(
            self, input_train, target_train, labels,
            **kwargs
    ):
        # Retrieve kwargs
        method = kwargs.get('method', 'sgd').lower()
        save_loss_progress = kwargs.get('save_loss_progress', None)
        save_output_progress = kwargs.get('save_output_progress', None)

        opt_options = kwargs.get(
            'options',
        )
        if opt_options is None:
            raise TypeError("Missing required keyword argument: 'opt_options'")

        lr = opt_options.get('lr')
        if lr is None:
            raise TypeError("Missing required argument in 'opt_options': 'lr'")

        lr_t = opt_options.pop('lr_t', lr)
        t_iter = opt_options.pop('t_iter', 1)

        optimizer = OPTIM_METHODS.get(method)(
            self, **opt_options
        )

        for it in range(self.__budget__):
            self.nbshots = self.nbshots_increment(
                self.nbshots, self.__n_iter__, self.__min_loss__)

            # optimize
            optimizer.step(
                input_train[self.__rnd_indices],
                target_train[self.__rnd_indices]
            )

            if it < t_iter:
                a = it / t_iter
                optimizer.set_lr(
                    (1 - a) * lr + a * lr_t
                )

            if save_loss_progress or save_output_progress:
                probas = self.predict_proba(
                    input_train[self.__rnd_indices],
                )
                loss_value = self.__loss__(
                    target_train[self.__rnd_indices],
                    probas,
                    labels=labels
                )

                self.__last_loss_value__ = loss_value
                self.__last_output__ = probas[np.argsort(self.__rnd_indices)]

            # Callback function
            self.__callback__(
                self.params, save_loss_progress, save_output_progress
            )

    def fit(self, input_train, target_train, batch_size=None,
            **kwargs):
        """Fit the model according to the given training data.

        Parameters
        ----------
        input_train : array
            Training design matrix.
        target_train : vector
            Labels corresponding to `input_train`.
        batch_size : int, optional
            Minibatches size, by default None. If none uses the full dataset
            with rndom shuffle at each iteration.
        method : str, optional
            Optimization method, by default BFGS
        bounds : sequence, optional
            Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
            trust-constr methods as a sequence of ``(min, max)`` pairs for
            each element in x. None is used to specify no bound.
        options : dict, optional
            Optimizer options, by default {'maxiter': budget}
        save_loss_progress : bool, optional
            Whether to store the loss progress, by default False
        save_output_progress : file path, optional
            Path where to save the output evolution , by default None. If none
            is given, the output is not saved.
        seed : int, optional
            Random seed, by default None

        Returns
        -------
        Classifier
            self
        """

        method = kwargs.get('method', 'BFGS').lower()
        save_output_progress = kwargs.get('save_output_progress', None)
        seed = kwargs.get('seed', None)

        if seed is not None:
            np.random.seed(seed)

        _nbshots = self.nbshots

        if batch_size:
            self.__batch_size = batch_size
        else:
            self.__batch_size = len(target_train)

        _labels = np.unique(target_train)
        if len(_labels) > len(self.bitstr):
            raise ValueError(
                f"Too many labels: expected {len(self.bitstr)}, found \
                {len(_labels)} in target_train"
            )

        self.__indices = np.arange(len(target_train))
        self.__rnd_indices = np.random.choice(
            self.__indices, size=self.__batch_size, replace=False
        )

        self.pbar = tqdm(total=self.__budget__, desc="Training", leave=False)
        self.__n_iter__ = 0

        if method in SCIPY_METHODS:
            self.__scipy_minimize__(
                input_train, target_train, _labels,
                **kwargs
            )
        elif method in OPTIM_METHODS.keys():
            self.__optim__(
                input_train, target_train, _labels,
                **kwargs
            )
        else:
            raise NotImplementedError

        self.pbar.close()
        del self.pbar
        if self.__n_iter__ < self.__budget__:
            if self.__fit_conv__:
                print(f"Early convergence at step {self.__n_iter__}")
            else:
                print(f"Optimization failed at step {self.__n_iter__}")

        if save_output_progress:
            with open(save_output_progress, "w") as f:
                _d = dict(output = self.__output_progress__,
                          labels = target_train.tolist(),
                          loss_value = self.__loss_progress__,
                          params = self.__params_progress__)
                json.dump(_d, f)
            self.__output_progress__ = []
            self.__params_progress__ = []

        # we reset the number if nbshots, as we changed it during training.
        self.nbshots = _nbshots

        return self

    def info_dict(self):
        """Returns a dictionary containing models information.

        Returns
        -------
        dict
            Information dictionary
        """
        out = {}

        model_info = {
            "parameters": self.params.tolist(),
            'circuit': str(self.circuit),
            'nbshots': self.nbshots,
            'nbshots_increment': str(self.nbshots_increment),
            'nbshots_incr_delay': str(self.nbshots_incr_delay),
            'bitstr': [bin(bit) for bit in self.bitstr],
            'job_size': self.job_size if self.job_size else "FULL",
            'nfev': self.nfev,
        }
        if self.__loss_progress__:
            model_info["loss_progress"] = self.__loss_progress__
        model_info["n_iter"] = self.__n_iter__

        name = "quantumClassifier"
        if self.__name__ is not None:
            name = self.__name__

        out[str(name)] = model_info
        return out

    def save(self, path=None):
        if path is None:
            path = self.__save_path__

        with open(path, 'wb') as f:
            pickle.dump(self.info_dict, f)
