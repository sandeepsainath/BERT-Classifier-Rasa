import logging
import warnings
import numpy as np
import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu import utils
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import MESSAGE_VECTOR_FEATURE_NAMES, MESSAGE_TEXT_ATTRIBUTE

class BertClassifier(Component):
    "A custom BERT classifier component using the bert-sklearn wrapper."

    provides = ['intent', 'intent_ranking']

    requires = [MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]]

    defaults = {
        # default model is bert-base-uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
        'bert_model': 'bert-base-uncased',
        # default is 128, max value is 512, halved for faster performance
        'max_seq_length': 64,
        # default is 32, halved for faster performance
        'train_batch_size': 16,
        # default is 3, can be reduced for faster performance
        'epochs': 3,
        # Fraction of the data used for validation, default is 0.1
        'validation_fraction': 0.1
    }

    language_list = ["en"]

    def __init__(
        self,
        component_config: Dict[Text, Any] = None,
        clf: "bert_sklearn.BertClassifier" = None,
        le: Optional["sklearn.preprocessing.LabelEncoder"] = None,
    ) -> None:
        """Construct a new intent classifier using the bert_sklearn framework."""
        from sklearn.preprocessing import LabelEncoder

        super().__init__(component_config)

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()

        self.clf = clf

    @classmethod
    def required_packages(cls):
        return ["sklearn", "bert_sklearn"]

    def transform_labels_str2num(self, labels: List[Text]) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.
        :param labels: List of labels to convert to numeric representation"""

        return self.le.fit_transform(labels)

    def transform_labels_num2str(self, y: np.ndarray) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.
        :param y: List of labels to convert to numeric representation"""

        return self.le.inverse_transform(y)

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any) -> None:
        """Train the intent classifier on a data set."""

        num_threads = kwargs.get("num_threads", 1)

        labels = [e.get("intent") for e in training_data.intent_examples]

        if len(set(labels)) < 2:
            warnings.warn(
                "Can not train an intent classifier. "
                "Need at least 2 different classes. "
                "Skipping training of intent classifier."
            )
        else:
            y = self.transform_labels_str2num(labels)
            X = np.stack(
                [
                    example.get("text_features")
                    for example in training_data.intent_examples
                ]
            )

            self.clf = self._create_classifier(num_threads, y)
            self.clf.fit(X, y)

    def _create_classifier(self, num_threads, y):
        from bert_sklearn import BertClassifier

        bert_model = self.component_config["bert_model"]
        epochs = self.component_config["epochs"]
        max_seq_length = self.component_config["max_seq_length"]
        train_batch_size = self.component_config["train_batch_size"]
        validation_fraction = self.component_config["validation_fraction"]

        return BertClassifier(
            bert_model=bert_model,
            epochs=epochs,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            validation_fraction=validation_fraction
        )

    def process(self, message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""

        if not self.clf:
            # component is either not trained or didn't
            # receive enough training data
            intent = None
            intent_ranking = []

        else:
            X = message.get("text_features").reshape(1, -1)
            intent_ids, probabilities = self.predict(X)
            intents = self.transform_labels_num2str(np.ravel(intent_ids))
            # `predict` returns a matrix as it is supposed
            # to work for multiple examples as well, hence we need to flatten
            probabilities = probabilities.flatten()

            if intents.size > 0 and probabilities.size > 0:
                ranking = list(zip(list(intents), list(probabilities)))[
                    :LABEL_RANKING_LENGTH
                ]

                intent = {"name": intents[0], "confidence": probabilities[0]}

                intent_ranking = [
                    {"name": intent_name, "confidence": score}
                    for intent_name, score in ranking
                ]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """Given a bow vector of an input text, predict the intent label.

        Return probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given a bow vector of an input text, predict most probable label.

        Return only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second,
                 its probability."""

        pred_result = self.predict_prob(X)
        # sort the probabilities retrieving the indices of
        # the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""

        classifier_file_name = file_name + "_classifier.pkl"
        encoder_file_name = file_name + "_encoder.pkl"
        if self.clf and self.le:
            utils.json_pickle(
                os.path.join(model_dir, encoder_file_name), self.le.classes_
            )
            utils.json_pickle(
                os.path.join(model_dir, classifier_file_name), self.clf
            )
        return {"classifier": classifier_file_name, "encoder": encoder_file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["BertClassifier"] = None,
        **kwargs: Any,
    ) -> "BertClassifier":
        from sklearn.preprocessing import LabelEncoder

        classifier_file = os.path.join(model_dir, meta.get("classifier"))
        encoder_file = os.path.join(model_dir, meta.get("encoder"))

        if os.path.exists(classifier_file):
            classifier = utils.json_unpickle(classifier_file)
            classes = utils.json_unpickle(encoder_file)
            encoder = LabelEncoder()
            encoder.classes_ = classes
            return cls(meta, classifier, encoder)
        else:
            return cls(meta)
