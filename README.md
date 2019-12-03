# BERT Classifier for Rasa NLU

This is Google's BERT Classifier implemented using the [bert-sklearn wrapper](https://github.com/charles9n/bert-sklearn) for a [Rasa chatbot](https://github.com/RasaHQ/rasa). This component is only compatible with any Python version above 3 and below Python 3.7. 

A default config.yml file for a Rasa project is included in this repo. It replaces the default classifier (also uses scikit-learn) in the pre-configured Rasa pipeline [`pretrained_embeddings_spacy`](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#section-pretrained-embeddings-spacy-pipeline) with this classifier.

The template of the component largely follows the same default scikit-learn intent classifier that uses Support Vector Machines (SVM) and GridSearchCV to optimize hyperparameters, but tweaked to implement the bert-sklearn wrapper as mentioned above. 

This version currently uses the smallest possible version of BERT for fastest performance. To tweak the type of BERT model and other hyperparameters, modify the `defaults` dictionary values defined right under the BertClassifier class definition.

**Note:** Your bash_profile's PYTHONPATH must be modified for Rasa to identify your custom component. More information on this is in the custom component tutorial linked below.

## Other Resources
* [Rasa Custom Component Tutorial](https://blog.rasa.com/enhancing-rasa-nlu-with-custom-components/)
* [Rasa Pipelines](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#section-pretrained-embeddings-spacy-pipeline)
