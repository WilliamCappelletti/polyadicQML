"""Module for auto-supervised clustering -- and classification.
"""

import numpy as np
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from .quantumModel import quantumModel, circuitML
from .utility import stable_softmax, CE_loss, CE_grad


class QMeans(quantumModel):
    def __init__(
        self,
        circuit: circuitML, nclasses: int,
        **kwargs
    ):
        super().__init__(circuit, **kwargs)

        # We randomly initialize the means
        # NOTE that is optional in the supervised setting

        self.means = stable_softmax(
            np.random.rand(nclasses, 2**circuit.nbqbits),
            axis=1
        )
        self.__v__ = 0
        self.__means_mov__ = []

    def dists(self, X: np.ndarray, params=None) -> np.ndarray:
        run_out = self.run_circuit(X, params)
        return cdist(run_out, self.means, metric="euclidean")

    def predict_proba(self, X: np.ndarray, params=None) -> np.ndarray:
        return self.run_out_to_proba(
            self.run_circuit(X, params)
        )

    def run_out_to_proba(self, run_out):
        return stable_softmax(
            - cdist(                    # NOTE negative distances
                run_out,
                self.means,
                metric="euclidean"
            ),
            axis=1
        )

    def __update_means__(self, run_out, targets):
        new_means = np.zeros_like(self.means)
        for i in range(len(self.means)):
            new_means[i] = np.mean(
                run_out[targets == i],
                axis=0
            )

        self.__means_mov__.append(
            np.linalg.norm(self.means - new_means)
        )

        self.means = new_means

    def __update_params__(self, probs, X, targets, **kwargs):
        momentum = kwargs.get("momentum", 0)
        lr = kwargs.get("lr", 0.1)
        n_steps = kwargs.get("n_steps", 1)

        N, *_ = probs.shape

        for step in range(n_steps):
            # If we do more than one iteration, we update the circuit outcome
            if step > 0:
                probs = self.predict_proba(X)

            # Compute partial gradient
            p_g = CE_grad(targets, probs)[:, :, 0] @ self.means
            # p_g = 2 / N * (probs - self.means[targets])

            G = self.circuit.grad(
                X, self.params,
                v=p_g,
                nbshots=self.nbshots,
                job_size=self.job_size,
            ).flatten()

            self.__v__ = momentum * self.__v__ - lr * G

            self.params += self.__v__

    def __update_assignments__(self, run_out):
        return np.argmin(
            cdist(run_out, self.means, metric="euclidean"),
            axis=1
        )

    def __stopping_criterion__(self):
        if np.allclose(
            self.__means_mov__[-3:], 0
        ):
            return True

        return False

    def fit(
        self, input_train, target_train = None, batch_size=None, **kwargs
    ):

        budget = kwargs.get("budget", self.__budget__)

        # We start by computing the means
        run_out = self.run_circuit(input_train)

        # If unsupervised update assignements
        if target_train is None:
            target_train = self.__update_assignments__(run_out)

        self.__update_means__(run_out, target_train)

        # Iterate b/w updating params, (assignements,) and means.
        for it in tqdm(range(budget)):

            self.__update_params__(
                # run_out,
                self.run_out_to_proba(run_out),
                input_train,
                target_train,
                **kwargs
            )

            # Compute new output from parameters
            run_out = self.run_circuit(input_train)

            # If unsupervised update assignements
            if target_train is None:
                target_train = self.__update_assignments__(run_out)

            self.__update_means__(run_out, target_train)

            if self.__stopping_criterion__():
                print(f"Stopping criterion met at iteration {it + 1}")
                break

        return self
